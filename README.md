# Task Scheduler using RL

An efficient and robust Task Scheduler that performs the best possible task scheduling while also judiciously managing resources for a given set of procedures or tasks, taken in as parameters, using a Deep Reinforcement Learning agent implemented using Keras-RL framework with the help of an environment built in Python using gym, which is based on the actions performed by the agent.

## Simulator

For the purpose of visual demo, we have designed a simulator, which takes in tasks in the form of percentage of CPU and Memory Utilisation, and displays the list of tasks scheduled using FCFS scheduling, and the agent along with the order of tasks.

### Libraries Used

Please install the following libraries by following their official installation documentation to run the simulator on your device.

* [tkinter](https://docs.python.org/3/library/tkinter.html)
* [numpy](https://numpy.org)
* [gym](https://pypi.org/project/gym/)
* [tensorflow](https://www.tensorflow.org/)
* [keras-rl](https://pypi.org/project/keras-rl/)

### Running the simulator

Once all the dependencies have been installed, run

```
python gui-simulator.py
```
