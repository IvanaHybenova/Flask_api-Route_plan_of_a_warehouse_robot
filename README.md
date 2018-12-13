# Flask_api-Route_plan_of_a_warehouse_robot
Flask api that returns optimal route plan of a warehouse robot (solved with Q-learning)

Business problem:

An online shop has a big warehouse, where a robot is picking ordered products. 

The warehouse has 25 locations (sections) and names are capital letters.

Goal of this project is to find the optimal route plan of a robot under the certain conditions:

1. Robot must enter the warehouse, go to the required location and return again to the entrance.

2. Robot can pick up products from the next required location as well, if it takes less steps than returning back from the first location
   and then go again to the next one.
   
3. If the robot passes through the entrance location on his way to the next required location, he stops in the entrance with product from the first location and route to the next required location is considered as a new one.
   
3. Robot can visit on one route through the warehouse only two required locations, so other processes don't have to wait too long for the    next product to be packed.


Project delivery: Python script executing locally hosted flask api, that takes in csv file with locations that needs to be visited, calculate the optimal route plan and provide downloadable zipped .xlsx file with the route plan.

Files: 

warehouse_plan.jpg - Picture of the warehouse and locations.

rewards.jpg - Rewards plan used for Q-learning. Thanks to this matrix robot basically knows, where it can go from current location.

flask_route_api.py - Python scirpt with the application

locations.csv - Locations robot needs to visit, in order to pick the products from them

Instructions: Download locations.csv and flask_route_api.py.

Through your command line navigate to the folder you are storing these files. Make sure you have python path in your enviroment variables and run command python flask_route_api.py

From your browser navigate to http://localhost:8000/apidocs. Click on predict_api and then try it out!. Insert locations.csv and press execute. Scroll down and click on Download the zip.file, which contains the predictions.

Go ahead and compute robots route plan of your own locations.csv file :)
