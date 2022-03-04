## Working Memory

---
- the glazing area ration component in the heat energy balance that is design specific.
  - it will need to be calculated as design and dimensions change 

- check to make sure RH to ABS humidity equations are accurate

- observation space will likely change
  - there is a high likelihood it contains redundant or unnecessary features
  - I should also explore how to scale the feature space to [0,1] or [-1,1]

- determine if I still need to have 25 indices in outside_radiative_heat and outside_RH
---
I have not kept a log book for this project but will begin now. 

## 2022.03.02

---
Discovering some bugs. 
Joy. 
The dynamics are not working exactly right. 
Need to check sensible and glazing losses.
They see extremely high. 

## 2022.03.03

---
Got both the humidity and the heat balance equations working. 
The issue was not with the equations or my programming but rather the ODE solver. 
Euler is insufficient for these and RK5 is needed. 

## 2022.03.04

---
Meeting with doug today. 
Main goal is to get a plot for ideal humidity vs time.
Time permitting, the next step is to humidity into the action space. 
For initial POC this will be actuated directly with a dehumidifier rather than affecting something indirect like ventilation.
