# Mini Auto-Gradient and Neural Network Software

## Introduction

Ever wondered how <b>`loss.backward()`</b> in popular frameworks like PyTorch manages to perform so much of the AI and neural network magic in just one line of code? Most practitioners use these functions daily, yet the intricate workings underneath remain a mystery to many. My fascination with the unseen led me to develop this mini auto gradient software, a journey into the core of neural network operations and gradient computing. This has a build in auto-grad, a neural network library, and PyTorch-like APIs.

![alt text](https://github.com/TejasKalsait/MicroGrad/blob/main/backprop.jpg)

## Features

- **Automatic Differentiation:** At the heart of this project is the auto grad feature, capable of efficiently computing gradients of complex functions with `scalar` values, similar to the backbone of neural network training in libraries like PyTorch.
- **Neural Network Library:** Built on top of the auto grad functionality, this component allows the creation and training of neural network models from scratch. It's designed to be intuitive yet flexible, catering to those who want to explore the creation of AI models without heavy reliance on pre-built frameworks.
- **Customizable Layers and Functions:** Every aspect of this library is designed with customization in mind. Users can define their own layers, activation functions, and more, gaining a deeper understanding of each component's role and behavior.

## Motivation

My motivation for this project stems from a relentless curiosity about the "how" and "why" behind the tools we use in AI and machine learning. By building this software from the ground up, I've gained a profound appreciation for the complexity and beauty of neural networks. This project is for those who share a similar curiosity and desire to peek behind the curtain of high-level APIs.

## Installation

```bash
git clone https://github.com/TejasKalsait/MiniTorch.git
cd mini_torch
```

## Demo

Go through the [`micrograd.ipynb`](https://github.com/TejasKalsait/MiniTorch/blob/main/micrograd.ipynb) file to understand the syntax of the engine. However, I have tried to keep it in sync with Pytorch.

## Notes

While building the software, I have taken some notes which are available in [`NOTES.md`](https://github.com/TejasKalsait/MiniTorch/blob/main/NOTES.md) file if anyone is interested.

## Reach out

Email - kalsaittejas10@gmail.com
[LinkedIn](https://www.linkedin.com/in/tkalsait/)  |  [Dreamer.ai](https://dreamer-ai.streamlit.app/)  |  [Portfolio](https://tejaskalsait.github.io/)
