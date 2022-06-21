# Human-level control through deep reinforcement learning
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Address neural network instability issues with:
- Experience replay (randomizes over the data and smoothes changes in distribution)
- Iterative play, only adjustcs Q periodically thereby reducing correlations with the target
- Store episode as D = {e_1,...e_t} where e_t = (s_t, a_t, r_t, s_(t + 1))
- Train on minibatches of experience (s,a,r,s') ~ U(D) drawn uniformly at random from pool of stored samples.

**Results**
- Performans comparaable to a pro human tester across 49 games

## Methods
Atari frames are 210x160, with 128-colour palette


### Preprocessing
- Encoding a single frame, take maximum value of pixel color value over the frame being encoded + previous frame
    - To combat flickering
-   Extract Y channel (luminance) from the RGB frame, and rescale to 84x84
- Algorithm 1: todo
    - Applies preprocessing to the `m` most recent frames and stacks them produce input to Q-function (m=4), although it works with other values for m (e.g. 3 and 5)
- Y channel + 4 stacked frames + rescaling means input is 84 x 84 x 4

#### Network:
- Input 84x84x4 
- 1st hidden layer colvolves 32 filters of 8x8 kernels with stride 4 and rectifier non-liear
- 2nd hidden layer colvoves 64 filters of 4x4 with stride 2 and rectifier non-linear
- 3rd hidden layer colvoves 64 filters of 3x3 with stride 1 and rectifier non-linear
- 4th hidden layer fully connected with 512 rectifier units
- Output layer fully connected linear layer with single output for each valid action (various from 4-18 depedning on game)

#### Training:
- 49 different Atari 2600 games
- Different network for each game, but all parameters are the same except output layer size
- Clip all rewards to range -1 to +1
    - Limit scale of derivates and makes use of same LR for different games easier
- For games with life counter, number of lives left in the game is used to mark end of an episode
- RMSProp optimizer , 0.95, 0.95
- Minibatch size of 32
- Episilon with linear annealing from 1.0 to 0.1 over 1m games (fixed at 0.1 therafter)
- Trained for 50 million frames (38 days of game experience)
- Replay memeory of 1 million most recent frames
- Frame skipping with k=4
    - (Agent only sees every kth frame instead of every frame to reduce computation)_
- Discount factor 0.99
- Learning rate = 0.00025
- Target network is only updated with Q network parameters every C steps
    - C = 10_000
- Experience replay: 1_000_00 (but should work with e.g. 10_000)
- Repeat action for 4 steps

#### Evaluation
- Play each game 30 times for 5 min with different initial random conditions (epsilon = 0.05)
- Baseline chose random action at 10Hz (every ~6th frame)


#### Algorithm
 TODO


