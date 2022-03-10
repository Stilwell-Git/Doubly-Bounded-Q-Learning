# Doubly Bounded Q-Learning through Abstracted Dynamic Programming (DB-ADP)

This is a TensorFlow implementation for our paper [On the Estimation Bias in Double Q-Learning](http://arxiv.org/abs/2109.14419) accepted by NeurIPS 2021.


## Requirements
1. Python 3.6.13
2. gym == 0.18.3
3. TensorFlow == 1.12.0
4. BeautifulTable == 0.8.0
5. opencv-python == 4.5.3.56

## Running Commands

Run the following commands to reproduce our main results shown in section 5.2.

```bash
python train.py --tag='DB-ADP Alien' --env=Alien
python train.py --tag='DB-ADP BankHeist' --env=BankHeist
python train.py --tag='DB-ADP BattleZone' --env=BattleZone
python train.py --tag='DB-ADP Frostbite' --env=Frostbite
python train.py --tag='DB-ADP Jamesbond' --env=Jamesbond
python train.py --tag='DB-ADP MsPacman' --env=MsPacman
python train.py --tag='DB-ADP Qbert' --env=Qbert
python train.py --tag='DB-ADP RoadRunner' --env=RoadRunner
python train.py --tag='DB-ADP StarGunner' --env=StarGunner
python train.py --tag='DB-ADP TimePilot' --env=TimePilot
python train.py --tag='DB-ADP WizardOfWor' --env=WizardOfWor
python train.py --tag='DB-ADP Zaxxon' --env=Zaxxon

python train.py --tag='DB-ADP-C Alien' --env=Alien --alg=cddqn
python train.py --tag='DB-ADP-C BankHeist' --env=BankHeist --alg=cddqn
python train.py --tag='DB-ADP-C BattleZone' --env=BattleZone --alg=cddqn
python train.py --tag='DB-ADP-C Frostbite' --env=Frostbite --alg=cddqn
python train.py --tag='DB-ADP-C Jamesbond' --env=Jamesbond --alg=cddqn
python train.py --tag='DB-ADP-C MsPacman' --env=MsPacman --alg=cddqn
python train.py --tag='DB-ADP-C Qbert' --env=Qbert --alg=cddqn
python train.py --tag='DB-ADP-C RoadRunner' --env=RoadRunner --alg=cddqn
python train.py --tag='DB-ADP-C StarGunner' --env=StarGunner --alg=cddqn
python train.py --tag='DB-ADP-C TimePilot' --env=TimePilot --alg=cddqn
python train.py --tag='DB-ADP-C WizardOfWor' --env=WizardOfWor --alg=cddqn
python train.py --tag='DB-ADP-C Zaxxon' --env=Zaxxon --alg=cddqn
```
