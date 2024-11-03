from typing import Any
import jax.numpy as jnp
import numpy as np
import tqdm
import jax
import flax
import optax
import wandb
import click
from utils.stable_vae import StableVAE
import matplotlib.pyplot as plt

from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from dit import DiT10M, DiT50M, DiT100M, DiTXL
from unet import UNet10M, UNet50M
from functools import partial
from imagenet import get_dataset
from shampoo import Shampoo
model_config = {
    'adam_lr': 1e-4,
    'adam_beta1': 0.9,
    'adam_beta2': 0.99,
    'shampoo_lr': 1e-4,
    'shampoo_beta1': 0.9,
    'shampoo_beta2': 0.99,

    'kron_lr': 7e-5,
    'kron_beta1': 0.9,
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 32,
    'cfg_scale': 4.0,
    'target_update_rate': 0.9999,
    't_sampler': 'log-normal',
    't_conditioning': 1,
}


# x_0 = Noise
# x_1 = Data
def get_x_t(images, eps, t):
    x_0 = eps
    x_1 = images
    t = jnp.clip(t, 0, 1 - 0.01)  # Always include a little bit of noise.
    return (1 - t) * x_0 + t * x_1

def get_v(images, eps):
    x_0 = eps
    x_1 = images
    return x_1 - x_0

class FlowTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    # Train
    @partial(jax.pmap, axis_name='data')
    def update(self, images, labels):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            # Sample a t for training.
            if self.config['t_sampler'] == 'log-normal':
                t = jax.random.normal(time_key, (images.shape[0],))
                t = (1 / (1 + jnp.exp(-t)))
            elif self.config['t_sampler'] == 'uniform':
                t = jax.random.uniform(time_key, (images.shape[0],), minval=0, maxval=1)

            t_full = t[:, None, None, None]  # [batch, 1, 1, 1]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)

            if self.config['t_conditioning'] == 0:
                t = jnp.zeros_like(t)

            v_prime = self.model.apply_fn({'params': params}, x_t, t, labels, train=True, rngs={'label_dropout': label_key})
            loss = jnp.mean((v_prime - v_t) ** 2)

            return loss, {
                'l2_loss': loss,
                'v_abs_mean': jnp.abs(v_t).mean(),
                'v_pred_abs_mean': jnp.abs(v_prime).mean(),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name='data')
        info = jax.lax.pmean(info, axis_name='data')

        updates, new_opt_state = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(step=self.model.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)

        # Update the model_eps
        new_model_eps = target_update(self.model, self.model_eps, 1 - self.config['target_update_rate'])
        if self.config['target_update_rate'] == 1:
            new_model_eps = new_model
        new_trainer = self.replace(rng=new_rng, model=new_model, model_eps=new_model_eps)
        return new_trainer, info

    @partial(jax.jit, static_argnames=('cfg'))
    def call_model(self, images, t, labels, cfg=True, cfg_val=1.0):
        if self.config['t_conditioning'] == 0:
            t = jnp.zeros_like(t)
        if not cfg:
            print(images.shape, t.shape, labels.shape)
            return self.model_eps.apply_fn({'params': self.model_eps.params}, images, t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * self.config['num_classes']  # Null token
            images_expanded = jnp.tile(images, (2, 1, 1, 1))  # (batch*2, h, w, c)
            t_expanded = jnp.tile(t, (2,))  # (batch*2,)
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            print(images_expanded.shape, t_expanded.shape, labels_full.shape)
            v_pred = self.model_eps.apply_fn({'params': self.model_eps.params}, images_expanded, t_expanded, labels_full, train=False, force_drop_ids=False)
            v_label = v_pred[:images.shape[0]]
            v_uncond = v_pred[images.shape[0]:]
            v = v_uncond + cfg_val * (v_label - v_uncond)
            return v

    @partial(jax.pmap, axis_name='data', in_axes=(0, 0, 0, 0, None, None), static_broadcasted_argnums=(4,5))
    def call_model_pmap(self, images, t, labels, cfg=True, cfg_val=1.0):
        return self.call_model(images, t, labels, cfg=cfg, cfg_val=cfg_val)


@click.command()
@click.option('--load_dir', default=None, help='Directory to load the model from.')
@click.option('--save_dir', default=None, help='Directory to save the model to.')
@click.option('--fid_stats', default=None, help='FID stats file.')
@click.option('--seed', default=None, type=int, help='Random seed.')
@click.option('--log_interval', default=1000, type=int, help='Logging interval.')
@click.option('--eval_interval', default=20000, type=int, help='Evaluation interval.')
@click.option('--save_interval', default=200000, type=int, help='Model saving interval.')
@click.option('--batch_size', default=256, type=int, help='Batch size.')
@click.option('--max_steps', default=100_000, type=int, help='Number of training steps.')
@click.option('--model_type', default='dit', help='dit or unet')
@click.option('--model_size', default=10, help='10 or 50')
@click.option('--scan_blocks', default=False, is_flag=True)
@click.option('--denoise_timesteps', default=100, help='Number of timesteps to denoise.')
@click.option('--_cfg_scale', default=2.0, help='Number of timesteps to denoise.')
@click.option('--psgd', default=False, is_flag=True)
@click.option('--repa', default=False, is_flag=True)
@click.option('--shampoo', default=False, is_flag=True)
def main(load_dir, save_dir, fid_stats, seed, log_interval, eval_interval, save_interval,
         batch_size, max_steps, model_type, model_size, scan_blocks, denoise_timesteps, 
         _cfg_scale, psgd, repa, shampoo):
    # jax distributed training setup
    jax.distributed.initialize()
    # Set default seed if not specified
    if seed is None:
        seed = np.random.choice(1000000)
    np.random.seed(seed)

    # Define model configuration

    # Preset configurations

    # Wandb configuration
    wandb_config = {}
    wandb_config.update({
        'project': 'flow',
        'name': f'{model_type}_{model_size}'+("_psgd" if psgd else '')+("_repa" if repa else '')
    })

    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = batch_size // (global_device_count // device_count)
    print("Global Batch Size: ", batch_size)
    print("Node Batch Size: ", local_batch_size)
    print("Device Batch Size:", local_batch_size // device_count)

    # Create WandB logger
    if jax.process_index() == 0:
        wandb.init(**wandb_config)

    # Data loading
    dataset = get_dataset(batch_size=local_batch_size, train=True)
    dataset_valid = get_dataset(batch_size=local_batch_size, train=False)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]

    rng = jax.random.PRNGKey(seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    vae = StableVAE.create()
    # example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
    vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
    vae_encode_pmap = jax.pmap(vae.encode)
    vae_decode = jax.jit(vae.decode)
    vae_decode_pmap = jax.pmap(vae.decode)

    # Model setup

    if model_type == 'dit':
        if model_size == 10:
            model_def = DiT10M(2, model_config["num_classes"], model_config['class_dropout_prob'], scan_blocks)
        elif model_size == 50:
            model_def = DiT50M(2, model_config["num_classes"], model_config['class_dropout_prob'], scan_blocks)
        elif model_size == 100:
            model_def = DiT100M(2, model_config["num_classes"], model_config['class_dropout_prob'], scan_blocks)
        elif model_size == 500:
            model_def = DiTXL(2, model_config["num_classes"], model_config['class_dropout_prob'], scan_blocks)
        else:
            raise ValueError("Invalid model size")
    elif model_type == 'unet':
        if model_size == 10:
            model_def = UNet10M(1, model_config["num_classes"], model_config['class_dropout_prob'])
        elif model_size == 50:
            model_def = UNet50M(1, model_config["num_classes"], model_config['class_dropout_prob'])
        else:
            raise ValueError("Invalid model size")

    example_t = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
    print("Total number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params)))

    if psgd:
        from psgd_jax.kron import kron
        if scan_blocks and model_type == 'dit':  # only dit supports scan rn
            all_false = jax.tree.map(lambda _: False, params)
            scanned_layers = flax.traverse_util.ModelParamTraversal(
                lambda p, _: "scan" in p or "Scan" in p
            ).update(lambda _: True, all_false)
        else:
            scanned_layers = None
        tx = kron(
            learning_rate=model_config["kron_lr"],
            b1=model_config["kron_beta1"],
            max_size_triangular=4000,
            trust_region_scale=1.5,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=True,  # useful for larger models that can't be vmapped all at once
            lax_map_batch_size=7,  # ideally should be a factor of depth (e.g. 7 for 28 layer net)
            # precond_update_precision="float32"
        )
    elif shampoo:
        tx = Shampoo(learning_rate=model_config["shampoo_lr"], beta1=model_config["shampoo_beta1"], beta2=model_config["shampoo_beta2"])
    else:
        tx = optax.adam(learning_rate=model_config['adam_lr'], b1=model_config['adam_beta1'], b2=model_config['adam_beta2'])

    model_ts = TrainState.create(model_def, params=params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params=params)
    model = FlowTrainer(rng, model_ts, model_ts_eps, model_config)

    if load_dir is not None:
        cp = Checkpoint(load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)
        del cp

    fid_stats = "/home/rami/dit-vs-unet/imagenet256_fidstats_openai.npz"

    if fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network()
        truth_fid_stats = np.load(fid_stats)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    model = model.replace(rng=jax.random.split(rng, len(jax.local_devices())))
    # Optional visualization (if needed)
    # jax.debug.visualize_array_sharding(model.model.params['FinalLayer_0']['Dense_0']['bias'])

    valid_images_small, valid_labels_small = next(dataset_valid)
    valid_images_small = valid_images_small[:device_count, None]
    valid_labels_small = valid_labels_small[:device_count, None]
    visualize_labels = example_labels.reshape((device_count, -1, *example_labels.shape[1:]))
    visualize_labels = visualize_labels[:, 0:1]
    imagenet_labels = open('./imagenet_labels.txt').read().splitlines()

    ###################################
    # Training Loop
    ###################################
    def eval_model():
        # Needs to be in a separate function so garbage collection works correctly.

        # Validation Losses
        valid_images, valid_labels = next(dataset_valid)
        valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
        valid_labels = valid_labels.reshape((len(jax.local_devices()), -1, *valid_labels.shape[1:]))
        _, valid_update_info = model.update(valid_images, valid_labels)
        valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
        valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
        if jax.process_index() == 0:
            wandb.log(valid_metrics, step=i)

        def process_img(img):
            img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img

        # Training loss on various t.
        mse_total = []
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((batch_images.shape), t)
            t_vector = jnp.full((batch_images.shape[0],batch_images.shape[1]), t)
            eps = jax.random.normal(key, batch_images.shape)
            x_t = get_x_t(batch_images, eps, t_full)
            v = get_v(batch_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, batch_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f'training_loss_t/{t}': mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({'training_loss_t/mean': mse_total.mean()}, step=i)

        # Validation loss on various t.
        mse_total = []
        fig, axs = plt.subplots(3, 10, figsize=(30, 20))
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((valid_images.shape), t)
            t_vector = jnp.full((valid_images.shape[0],batch_images.shape[1]), t)
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t_full)
            v = get_v(valid_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, valid_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f'validation_loss_t/{t}': mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({'validation_loss_t/mean': mse_total.mean()}, step=i)
            plt.close(fig)

        # One-step denoising at various noise levels.
        # This only works on a TPU node with 8 devices for now...
        if len(jax.local_devices()) == 4:
            # assert valid_images.shape[0] == len(jax.local_devices()) # [devices, batch//devices, etc..]
            t = jnp.arange(4) / 4 # between 0 and 0.875
            t = jnp.repeat(t[:, None], valid_images.shape[1], axis=1) # [8, batch//devices, etc..] DEVICES=8
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t[..., None, None, None])
            v_pred = model.call_model_pmap(x_t, t, valid_labels, False, 0.0)
            x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(4, 4*3, figsize=(90, 30))
                for j in range(4):
                    for k in range(4):
                        axs[j,3*k].imshow(process_img(valid_images[j,k]), vmin=0, vmax=1)
                        axs[j,3*k+1].imshow(process_img(x_t[j,k]), vmin=0, vmax=1)
                        axs[j,3*k+2].imshow(process_img(x_1_pred[j,k]), vmin=0, vmax=1)
                wandb.log({f'reconstruction_n': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # Full Denoising with different CFG;
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape) # [devices, batch//devices, etc..]
        delta_t = 1.0 / denoise_timesteps
        for cfg_scale in [0, 0.1, 1, 4, 10]:
            x = eps
            all_x = []
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                v = model.call_model_pmap(x, t_vector, visualize_labels, True, cfg_scale)
                x = x + v * delta_t
                if ti % (denoise_timesteps // 8) == 0 or ti == denoise_timesteps-1:
                    all_x.append(np.array(x))
            all_x = np.stack(all_x, axis=2) # [devices, batch//devices, timesteps, etc..]
            all_x = all_x[:, :, -4:]

            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(4, 4, figsize=(30, 30))
                for j in range(4):
                    for t in range(4):
                        axs[t, j].imshow(process_img(all_x[j, 0, t]), vmin=0, vmax=1)
                    axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
                wandb.log({f'sample_cfg_{cfg_scale}': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # Denoising at different numbers of steps.
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape) # [devices, batch//devices, etc..]
        delta_t = 1.0 / denoise_timesteps
        for _denoise_timesteps in [1, 4, 32]:
            x = eps
            all_x = []
            for ti in range(_denoise_timesteps):
                t = ti / _denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                v = model.call_model_pmap(x, t_vector, visualize_labels, True, _cfg_scale)
                x = x + v * delta_t
            if jax.process_index() == 0:
                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(8):
                        axs[t, j].imshow(process_img(x[j, t]), vmin=0, vmax=1)
                    axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
                wandb.log({f'sample_N/{denoise_timesteps}': wandb.Image(fig)}, step=i)
                plt.close(fig)

        # FID calculation.
        activations = []
        valid_images_shape = valid_images.shape
        for fid_it in range(4096 // batch_size):
            _, valid_labels = next(dataset_valid)
            valid_labels = valid_labels.reshape((len(jax.local_devices()), -1, *valid_labels.shape[1:]))

            key = jax.random.PRNGKey(42 + fid_it)
            x = jax.random.normal(key, valid_images_shape)
            delta_t = 1.0 / denoise_timesteps
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                if _cfg_scale == -1:
                    v = model.call_model_pmap(x, t_vector, valid_labels, False, 0.0)
                else:
                    v = model.call_model_pmap(x, t_vector, valid_labels, True, _cfg_scale)
                x = x + v * delta_t
            x = vae_decode_pmap(x)
            x = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
            x = 2 * x - 1
            acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
            acts = jax.pmap(lambda x: jax.lax.all_gather(x, 'i', axis=0), axis_name='i')(acts)[0] # [global_devices, batch//global_devices, 2048]
            acts = np.array(acts)
            activations.append(acts)
        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            wandb.log({'fid': fid}, step=i)

        del valid_images, valid_labels
        del all_x, x, x_t, eps
        print("Finished all the eval stuff")

    for i in tqdm.tqdm(range(0, max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch_images, batch_labels = next(dataset)
        batch_images = batch_images.reshape((len(jax.local_devices()), -1, *batch_images.shape[1:]))  # [devices, batch//devices, etc..]
        batch_labels = batch_labels.reshape((len(jax.local_devices()), -1, *batch_labels.shape[1:]))

        model, update_info = model.update(batch_images, batch_labels)

        if i % log_interval == 0:
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % eval_interval == 0 or i == 1000:
            eval_model()

        if i % save_interval == 0 and save_dir is not None:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(save_dir, parallel=False)
                cp.set_model(model_single)
                cp.save()
                del cp, model_single

if __name__ == '__main__':
    main()
