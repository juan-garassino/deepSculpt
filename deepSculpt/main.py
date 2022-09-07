from deepSculpt.manager.trainer import trainer


def main(new_train=False, new_data=False, data='', epochs='' ,local='', colab='', gcp=''):
    trainer(new_train, new_data, data, epochs, local, colab, gcp)
