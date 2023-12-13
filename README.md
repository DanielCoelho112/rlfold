# RLOLD
This is the implementation of RLfOLD, which is described in:

> **RLfOLD: Reinforcement Learning from Online Demonstrations in Urban Autonomous Driving**
>
> [Daniel Coelho](https://github.com/DanielCoelho112), 
[Miguel Oliveira](https://github.com/miguelriemoliveira),
[Vítor Santos](https://github.com/vitoruapt).
>
> [AAAI 2024](https://aaai.org/aaai-conference/)<br/>

If you find our work useful, please consider citing: WIP
<!-- ```bibtex
@inproceedings{mile2022,
  title     = {Model-Based Imitation Learning for Urban Driving},
  author    = {Anthony Hu and Gianluca Corrado and Nicolas Griffiths and Zak Murez and Corina Gurau
   and Hudson Yeo and Alex Kendall and Roberto Cipolla and Jamie Shotton},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  year = {2022}
} -->

## Setup
- Clone the repository with `git clone git@github.com:DanielCoelho112/rlfold.git`
- Download [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1).
- Run the docker container with `docker run -it --gpus all --network=host -v results_path:/root/results/rlfold -v rlfold_path:/root/rlfold danielc11/rlfold:0.0 bash`
where `results_path` is the path where the results will be written, and `rlfold_path` is the path of the rlfold repository.


## Training
- Start the CARLA server
- Run: `python3 rlfold/run/python3 main.py -en rlfold_original`


## Credits
Thanks to the authors of [End-to-End Urban Driving by Imitating a Reinforcement Learning Coach](https://github.com/zhejz/carla-roach)
for providing a framework to train RL agent in CARLA.
