{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Tensorflow.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOq5AZGnAx8Pk1v3vExuBe1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rohitpawar4507/Tensorflow-Image-Classification/blob/main/Tensorflow.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2pen1w29ELmx"
      },
      "source": [
        "import tensorflow as tf"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aZbCVbWHEQBR",
        "outputId": "b965343b-ed51-46a9-b40f-321b3dc0f99e"
      },
      "source": [
        "print(tf.__version__ )"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "2.5.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0fQpeBNXNdVx",
        "outputId": "07aab2a9-bc56-4cc1-8e3c-4dbff6447029"
      },
      "source": [
        "import tensorflow.keras\n",
        "from PIL import Image, ImageOps\n",
        "import numpy as np\n",
        "\n",
        "# Disable scientific notation for clarity\n",
        "np.set_printoptions(suppress=True)\n",
        "\n",
        "# Load the model\n",
        "model = tensorflow.keras.models.load_model('keras_model.h5')\n",
        "\n",
        "# Create the array of the right shape to feed into the keras model\n",
        "# The 'length' or number of images you can put into the array is\n",
        "# determined by the first position in the shape tuple, in this case 1.\n",
        "data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)\n",
        "\n",
        "# Replace this with the path to your image\n",
        "image = Image.open('test_photo.jpg')\n",
        "\n",
        "#resize the image to a 224x224 with the same strategy as in TM2:\n",
        "#resizing the image to be at least 224x224 and then cropping from the center\n",
        "size = (224, 224)\n",
        "image = ImageOps.fit(image, size, Image.ANTIALIAS)\n",
        "\n",
        "#turn the image into a numpy array\n",
        "image_array = np.asarray(image)\n",
        "\n",
        "# display the resized image\n",
        "image.show()\n",
        "\n",
        "# Normalize the image\n",
        "normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1\n",
        "\n",
        "# Load the image into the array\n",
        "data[0] = normalized_image_array\n",
        "\n",
        "# run the inference\n",
        "prediction = model.predict(data)\n",
        "print(prediction)\n"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
            "[[0.1959461  0.80405396]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7r-o7Rt1PEfY"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tx3edHsjOc1S"
      },
      "source": [
        "# New Section"
      ]
    }
  ]
}