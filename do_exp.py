import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
import numpy as np
import utils
import MDN as mdn
from resnets_utils import TDResNet50

from tensorflow.python.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
tf.autograph.set_verbosity(3)


def main(args):

    with tf.device(f"/GPU:{args.device}"):

        exp_name = f"{args.exp_name}_" \
                   f"sub:{args.sub_id}_" \
                   f"{args.model}_" \
                   f"x:{args.exp_x}_" \
                   f"y:{args.exp_y}" \
                   f"+{'abs(res)' if args.absres else 'res'}_" \
                   f"ctx:{args.ctx_pre}-{args.ctx_post}_" \
                   f"l1:{args.reg}" \
                   f"l2:{args.reg2}"

        print(f"Doing Experiment {exp_name}")
        print("Preparing data...")
        ctx = (args.ctx_pre, args.ctx_post)

        split = args.split
        sub_id = args.sub_id
        detector = args.detector

        xx, yy, le = utils.prepare_dataset(sub_id, ctx, split, args.exp_x, args.exp_y, args.absres)
        
        print("Done!")
        print("Preparing model")
        exp_x = args.exp_x
        exp_y = args.exp_y

        opt = tf.optimizers.Adam(args.lr)

        tb_writer = tf.summary.create_file_writer(f'tbdir/{exp_name}')

        checkpoint_path = f"./models/{exp_name}" + "/cp-{epoch:04d}.ckpt"

        cp_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True)

        y_train = yy['train'][exp_y]
        y_test = yy['test'][exp_y]

        x_train = xx['train'][exp_x]
        x_test = xx['test'][exp_x]

        input_shape = x_train.shape[1:]
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {y_train.shape[1:]}")
        print(f"Using Device: /GPU:{args.device}")

        if args.detector is None:
            model = TDResNet50(input_shape=input_shape, components=args.components,
                               reg=args.reg, num_detectors=10, gauss_noise=args.gauss, reg2=args.reg2)
            print(model.summary())
        else:
            model = TDResNet50(input_shape=input_shape, components=args.components,
                               reg=args.reg, num_detectors=1, gauss_noise=args.gauss)
        print(f"Model ready with {model.count_params()} params")

        train_loader = utils.DataGeneratorMixup(x_train,
                                                y_train,
                                                n_samples=10000,
                                                mixup=args.mixup,
                                                batch_size=args.batch_size,
                                                det=args.detector)

        val_loader = utils.DataGeneratorMixup(x_test,
                                              y_test,
                                              shuffle=False,
                                              det=args.detector)

        if args.detector is not None:
            y_train = y_train[:, args.detector - 1][:, None]
            y_test = y_test[:, args.detector - 1][:, None]

        if args.detector is None:
            print(f"LOSS WEIGHTS  {[np.mean(__y) for __y in y_train.T]}")
            model.compile(loss=[mdn.gnll_loss for _ in range(len(y_train.T))],
                          loss_weights=[np.mean(__y) for __y in y_train.T],
                          optimizer=opt,
                          metrics=[mdn.mdn_eval_corr])
        else:
            model.compile(loss=[mdn.gnll_loss], optimizer=opt,
                          metrics=[mdn.mdn_eval_corr, mdn.mdn_eval_corr_map])
        # Will do 1 epoch at a time to have a full correlation in the end
        with tb_writer.as_default():
            for epoch in range(args.epochs):
                model.fit(train_loader, epochs=epoch + 1, validation_data=val_loader,
                          callbacks=[cp_callback], verbose=1, initial_epoch=epoch)

                gm_pred_train = model.predict(x_train)
                gm_pred_test = model.predict(x_test)

                if len(gm_pred_train) != 10:
                    gm_pred_train = [gm_pred_train]
                    gm_pred_test = [gm_pred_test]

                tr_mse, tr_nll, tr_corr, tr_corr_map = 0., 0., 0., 0.
                te_mse, te_nll, te_corr, te_corr_map = 0., 0., 0., 0.

                for _pred, _true in zip(gm_pred_train, y_train.T):
                    _tr_mse, _tr_nll, _tr_corr, _tr_corr_map = mdn.eval_mdn_model(_pred, _true[:, None])
                    tr_mse += _tr_mse
                    tr_nll += _tr_nll
                    tr_corr += _tr_corr
                    tr_corr_map += _tr_corr_map

                for _pred, _true in zip(gm_pred_test, y_test.T):
                    _te_mse, _te_nll, _te_corr, _te_corr_map = mdn.eval_mdn_model(_pred, _true[:, None])
                    te_mse += _te_mse
                    te_nll += _te_nll
                    te_corr += _te_corr
                    te_corr_map += _te_corr_map

                tf.summary.scalar("tr_mse", tr_mse / len(gm_pred_test), step=epoch)
                tf.summary.scalar("tr_nll", tr_nll / len(gm_pred_test), step=epoch)
                tf.summary.scalar("tr_corr", tr_corr / len(gm_pred_test), step=epoch)
                tf.summary.scalar("tr_corr_map", tr_corr_map / len(gm_pred_test), step=epoch)
                tf.summary.scalar("te_mse", te_mse / len(gm_pred_test), step=epoch)
                tf.summary.scalar("te_nll", te_nll / len(gm_pred_test), step=epoch)
                tf.summary.scalar("te_corr", te_corr / len(gm_pred_test), step=epoch)
                tf.summary.scalar("te_corr_map", te_corr_map / len(gm_pred_test), step=epoch)
                tb_writer.flush()

                print(f"Train NLL: {tr_nll / len(gm_pred_test):.4f}")
                print(f"Train COR: {tr_corr / len(gm_pred_test):.4f}")
                print(f"Val NLL: {te_nll / len(gm_pred_test):.4f}")
                print(f"Val COR: {te_corr / len(gm_pred_test):.4f}")

                # Plot some stuff
                for i, (_y_train, _y_test, _gm_pred_train, _gm_pred_test) in enumerate(zip(y_train.T,
                                                                                           y_test.T,
                                                                                           gm_pred_train,
                                                                                           gm_pred_test)):

                    alpha_pred, mu_pred, sigma_pred = mdn.slice_parameter_vectors(_gm_pred_train, 3, 3)
                    y_pred_train = np.multiply(alpha_pred, mu_pred).sum(axis=-1)
                    y_pred_train_map = np.array([mu_pred[i, k] for i, k in enumerate(np.argmax(alpha_pred, axis=1))])

                    alpha_pred, mu_pred, sigma_pred = mdn.slice_parameter_vectors(_gm_pred_test, 3, 3)
                    y_pred_test = np.multiply(alpha_pred, mu_pred).sum(axis=-1)
                    y_pred_test_map = np.array([mu_pred[i, k] for i, k in enumerate(np.argmax(alpha_pred, axis=1))])

                    fig, ax = plt.subplots(2, 1, figsize=(20, 5))
                    ax[0].plot(_y_train[:300], label='true')
                    ax[0].plot(y_pred_train[:300], label='pred')
                    ax[0].plot(y_pred_train_map[:300], label='pred-MAP')
                    ax[1].plot(_y_test[:300], label='true')
                    ax[1].plot(y_pred_test[:300], label='pred')
                    ax[1].plot(y_pred_test_map[:300], label='pred-MAP')
                    ax[0].legend()
                    ax[1].legend()

                    tf.summary.image(f"Best prediction det {i}", utils.plot_to_image(fig), step=epoch)

                    fig, ax = plt.subplots(2, 1, figsize=(20, 5))
                    alpha, mu, sigma = mdn.slice_parameter_vectors(_gm_pred_test, 3, 3)
                    gm = tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=alpha),
                        components_distribution=tfd.Normal(
                            loc=mu,
                            scale=sigma))
                    x = np.tile(np.linspace(0, 1, int(1e3)), (_gm_pred_test.shape[0], 1)).T
                    pyx = gm.prob(x)

                    ax[0].imshow(pyx[:, :100], aspect='auto')
                    ax[0].plot(_y_test.squeeze()[:100] * 1000, '--r')
                    ax[0].invert_yaxis()
                    ax[1].plot(pyx[:, :100])

                    tf.summary.image(f"Predicted distribution det {i}", utils.plot_to_image(fig), step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='le-net')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--mixup', type=str, default='')
    parser.add_argument('--absres', action='store_true', default=False)
    parser.add_argument('--gauss', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--reg', type=float, default=1e-3)
    parser.add_argument('--reg2', type=float, default=1e-3)
    parser.add_argument('--detector', type=int, default=None)
    parser.add_argument('--sub-id', type=str, default='1')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--ctx-pre', type=int, default=4)
    parser.add_argument('--ctx-post', type=int, default=4)
    parser.add_argument('--n-param', type=int, default=3)
    parser.add_argument('--components', type=int, default=3)

    parser.add_argument('--exp-x', type=str, default='img')
    parser.add_argument('--exp-y', type=str, default='full')

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--exp-name', type=str, default='le_netr_default')

    arguments, _ = parser.parse_known_args()

    main(arguments)
