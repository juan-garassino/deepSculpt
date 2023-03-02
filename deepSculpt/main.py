from deepSculpt.manager.manager import Manager
from deepSculpt.trainer.trainerbackup import trainer
from deepSculpt.curator.curator import Curator
from deepSculpt.trainer.tools.simple_model import (
    make_three_dimentional_generator,
    make_three_dimentional_critic,
)
from colorama import Fore, Style
import os
from google.cloud import storage
from tensorflow.train import Checkpoint, CheckpointManager
from deepSculpt.trainer.tools.optimizers import (
    generator_optimizer,
    discriminator_optimizer,
)
from tensorflow import GradientTape, function, Variable


def main(create_data=0, collection_folder=".", epochs=5, monochrome=0):

    # LOADS DATA
    if create_data == 0:
        pass

    # CREATES DATA
    if create_data == 1:
        pass

    # BROKEN
    else:
        print("BROKEN")

    # MONOCHROME SETUP
    if monochrome == 0:
        pass

    # COLOR SETUP
    if monochrome == 1:

        # Loads and process data
        curator = Curator(processing_method="OHE")

        # Initiates the Generator

        generator = make_three_dimentional_generator()

        generator.compile()

        print("\n ❎ " + Fore.RED + "The Generators summary is" + Fore.YELLOW + "\n ")

        # print(generator.summary())

        # Initiates the Discriminator

        discriminator = make_three_dimentional_critic()

        discriminator.compile()

        print(
            "\n ❎ " + Fore.RED + "The Discriminators summary is" + Fore.YELLOW + "\n "
        )

        # print(discriminator.summary())

        print(Style.RESET_ALL)

        ## Path local instance enviroment on COMPUTER

        if int(os.environ.get("INSTANCE")) == 0:

            checkpoint_dir = os.path.join(
                os.environ.get("HOME"),
                "code",
                "juan-garassino",
                "deepSculpt",
                "results",
                "checkpoints",
            )

            Manager.make_directory(checkpoint_dir)

        ## Path local instance eviroment on COLAB

        if int(os.environ.get("INSTANCE")) == 1:

            checkpoint_dir = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "drive",
                "MyDrive",
                "repositories",
                "deepSculpt",
                "results",
                "checkpoints",
            )

            Manager.make_directory(checkpoint_dir)

        ## Path cloud instance enviroment in GCP

        if int(os.environ.get("INSTANCE")) == 2:

            checkpoint_dir = "gs://deepsculpt/checkpoints"

            bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

        # Initiates a Checkpoint Object

        checkpoint = Checkpoint(
            step=Variable(1),
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

        # Initiates a Checkpoint Manager Object

        manager = CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
            checkpoint_name="checkpoint",
        )

    # BROKEN
    else:
        print("BROKEN")

    # TRAIN LOOP
    trainer(collection_folder, curator, epochs)


if __name__ == "__main__":

    collection_folder = os.path.join(
        os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
    )

    main(
        collection_folder,
        int(os.environ.get("EPOCHS")),
        monochrome=int(os.environ.get("COLOR")),
    )
