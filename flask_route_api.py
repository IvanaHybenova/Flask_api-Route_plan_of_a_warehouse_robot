"""
Created on Fri Mar 16 21:06:35 2018

@author: Ivana Hybenoa
"""

from flask import Flask, request, make_response, send_file
from flasgger import Swagger
import numpy as np
import pandas as pd
import zipfile
import time
from io import BytesIO

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: file_input_test
        in: formData
        type: file
        required: true
    responses:
        200:
            description: OK
    """
    dataset = pd.read_csv(request.files.get("file_input_test"))

    gamma = 0.75  # the discount factor in the temporal difference
    alpha = 0.9 # the learning rate
    
    # PART 1 - DEFINING THE ENVIROMENT
    
    # Defining the states
    location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                         'ENTRANCE':5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
                         'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14,
                         'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19,
                         'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24}
    
    # Defining the rewards
    R = np.array([[0,1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [1,	0, 0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0, 0,	0,	0],
                  [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0, 0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0, 0, 	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	1,	0],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	1],
                  [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0]])
    
    # PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING
    
    #Maing a mapping from the states to the locations
    state_to_location  = {state: location for location, state in location_to_state.items()}
    
    # Making the  function that will return the optimal route   
    def route(starting_location, ending_location):
        R_new = np.copy(R)
        ending_state = location_to_state[ending_location]
        R_new[ending_state,ending_state] = 1000
        
        # Initializing the Q-Values
        Q = np.array(np.zeros([25,25]))
        # Implementing the Q-Learning process
        for i in range(1000):                      # just 1000 because range goes from 0 by default
            current_state = np.random.randint(0,25)
            playable_actions = []
            for j in range(25):
                if R_new[current_state, j] > 0:
                    playable_actions.append(j)
            next_state = np.random.choice(playable_actions)
            TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
            Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
     
        route = [starting_location]
        next_location = starting_location
        while (next_location != ending_location):
            starting_state = location_to_state[starting_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = state_to_location[next_state]
            route.append(next_location)
            starting_location = next_location
        return route
    
    # PART 3 - GOING INTO PRODUCTION
    
    locations = dataset.iloc[:, 0:1].values.squeeze()
    entrance = 'ENTRANCE'
    def decided_route(required_location, considered_location):
        required_route = route(entrance, required_location) + route(required_location, entrance)[1:]
        next_route = route(entrance, considered_location) + route(considered_location, entrance)[1:]
        considered_route = route(entrance, required_location) + route(required_location, considered_location)[1:] + route(considered_location, entrance)[1:]
        considered_route2 = route(entrance, considered_location) + route(considered_location, required_location)[1:] + route(required_location, entrance)[1:]
        if (len(considered_route) > (len(required_route) + len(next_route)) and
            len(considered_route2) > (len(required_route) + len(next_route))):
            return required_route
       
        if (len(considered_route) <= (len(required_route) + len(next_route))):
            passed_entrance = [location for location in considered_route if 'ENTRANCE' in location]
            if(len(passed_entrance) == 2):
                return considered_route
            else:
                return required_route
        
        if (len(considered_route2) <= (len(required_route) + len(next_route))):
            passed_entrance = [location for location in considered_route if 'ENTRANCE' in location]
            if(len(passed_entrance) == 2):
                return considered_route2
            else:
                return required_route
            
    def last_route(required_location):
        return route(entrance, required_location) + route(required_location, entrance)[1:]
      
    done = []
    i = 0
    output = [['Locations robot is going to visit:', locations]]
    output.append([''])
    while i < len(locations):   
        j = i + 1
        required_location = locations[i]
        output.append(['Next required location:' , required_location])
        if i == (len(locations) - 1):
            picked_route = last_route(required_location)
            done.append(required_location)
            output.append(['Picked route:' , picked_route])
            output.append(['Visited locations:' , required_location])
        else:
            considered_location = locations[j]
            output.append(['Next considered location:' , considered_location])
            picked_route = decided_route(required_location, considered_location)
            output.append(['Picked route:' , picked_route])

            
            if considered_location in picked_route:
               done.append(required_location)
               done.append(considered_location)
               output.append(['Visited locations:' , required_location + ',' + considered_location])
            else:
               done.append(required_location)
               output.append(['Visited locations:' , required_location])
        i = len(done)
        output.append([''])
        output.append(['Visited locations after all routes:' , done])

    data = pd.DataFrame(output)
   
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='route_plan', 
                        encoding='utf-8', index=False, header = None)
    writer.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['route_plan.xlsx'] # names = ['file1.xlsx', 'file2']
        files = [output]  # files = [output, output2]
        for i in range(len(files)):
            input_data = zipfile.ZipInfo(names[i])
            input_data.date_time = time.localtime(time.time())[:6]
            input_data_compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(input_data, files[i].getvalue())
    memory_file.seek(0)
    
    response = make_response(send_file(memory_file, attachment_filename = 'route_plan.zip',
                                       as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment;filename=route_plan.zip'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    