import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from huggingface_hub import hf_hub_download

from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset

def create_bridge_features():
    """Create the FeaturesDict matching the original Bridge dataset structure."""
    return tfds.features.FeaturesDict({
        'steps': tfds.features.Dataset({
            'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            'language_embedding': tfds.features.Tensor(shape=(512,), dtype=tf.float32),
            'is_terminal': tfds.features.Scalar(dtype=tf.bool),
            'is_last': tfds.features.Scalar(dtype=tf.bool),
            'language_instruction': tfds.features.Text(),
            'observation': tfds.features.FeaturesDict({
                'image_0': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'image_1': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'image_2': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'image_3': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'state': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            }),
            'is_first': tfds.features.Scalar(dtype=tf.bool),
            'discount': tfds.features.Scalar(dtype=tf.float32),
            'reward': tfds.features.Scalar(dtype=tf.float32),
        }),
        'episode_metadata': tfds.features.FeaturesDict({
            'has_image_0': tfds.features.Scalar(dtype=tf.bool),
            'has_image_1': tfds.features.Scalar(dtype=tf.bool),
            'has_image_2': tfds.features.Scalar(dtype=tf.bool),
            'has_image_3': tfds.features.Scalar(dtype=tf.bool),
            'has_language': tfds.features.Scalar(dtype=tf.bool),
            'file_path': tfds.features.Text(),
            'episode_id': tfds.features.Scalar(dtype=tf.int32),
        }),
    })

if __name__ == "__main__":
    # Use builder_from_directory to load the dataset directly from the directory
    builder = tfds.builder_from_directory(
        builder_dir="/srv/rl2-lab/flash7/rbansal66/embodied-CoT/data/bridge_orig/1.0.0"
    )
    dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)  # Use "val" since that's what you have
    
    # Use TensorFlow string operations instead of Python 'in' operator
    def filter_instruction(episode):
        return tf.strings.regex_full_match(
            episode["language_instruction"], 
            b".*move the brown toy to the lower right burner.*"
        )
    
    dataset = dataset.unbatch().filter(filter_instruction)
    
    custom_data_dir = "/srv/rl2-lab/flash7/rbansal66/embodied-CoT/trunc_data"

    bridge_features = create_bridge_features()

    data_builder = tfds.dataset_builders.TfDataBuilder(
        name="bridge_dataset",
        config="bridge_orig",
        version="1.0.0",
        data_dir=custom_data_dir,
        split_datasets={
            "train": dataset,
        },
        # features=None
        features=bridge_features,
    )
    data_builder.download_and_prepare()