# CartPole (v0)

This is the folder containing the code for Cartpole-v0 Reinforcement Learning project

## Requirements

This project requires:
 * [OpenAI Gym environment](https://github.com/openai/gym)
 * [Tensorflow](https://www.tensorflow.org)
 * ArgParse

### Installing

#### OpenAI Gym

```
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
```
The minimal install is fine for the CartPole project

#### ArgParse

```
pip install argparse
```
#### Tensorflow

Refer to the [instructions](https://www.tensorflow.org/install/) on the Tensorflow website.


## Running

To run:

```
python cartpole-v0.py --arguments-to-pass
```
The only required argument is the *--action-choice* argument, which specifies the
transition model to use. The different options are:
 * *random*: Randomly choose parameters for each iteration
 * *hill-climb*: Randomly initialize parameters, add a small amount of noise to the parameters, and evaluate the new parameter configuration
 * *policy-gradient*: The action is chosen randomly where the distribution over actions depends on the numerical output of the inner product
