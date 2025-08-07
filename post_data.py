import requests

# Define the URL of the API
url = "http://localhost:5000/points"

# Create a dictionary with the data you want to send
data = {
    "poses": [{
        "joints":    {"right_upLeg_joint": {"x": 0.7095624208450317, "y": 0.3953590989112854, "confidence": 0.76171875}, "right_forearm_joint": {"x": 0.6865921020507812, "y": 0.44918370246887207, "confidence": 0.7993164}, "left_leg_joint": {"x": 0.765120267868042, "y": 0.24912726879119873, "confidence": 0.765625}, "left_hand_joint": {"x": 0.802729606628418, "y": 0.37088263034820557, "confidence": 0.8852539}, "left_ear_joint": {"x": 0.7587020397186279, "y": 0.6000765562057495, "confidence": 0.6279297}, "left_forearm_joint": {"x": 0.7896426320075989, "y": 0.4499925374984741, "confidence": 0.8852539}, "right_leg_joint": {"x": 0.7082920074462891, "y": 0.2478961944580078, "confidence": 0.83447266}, "right_foot_joint": {"x": 0.7081795334815979, "y": 0.12643909454345703, "confidence": 0.8178711}, "right_shoulder_1_joint": {"x": 0.6981396675109863, "y": 0.535271406173706, "confidence": 0.87841797}, "neck_1_joint": {"x": 0.7372709512710571, "y": 0.5358762741088867, "confidence": 0.8857422}, "left_upLeg_joint": {"x": 0.7619205117225647, "y": 0.3969404697418213, "confidence": 0.75097656}, "left_foot_joint": {"x": 0.7669140100479126, "y": 0.11862164735794067, "confidence": 0.85791016}, "root": {"x": 0.7357414662837982, "y": 0.39614978432655334, "confidence": 0.75634766}, "right_hand_joint": {"x": 0.6731472015380859, "y": 0.3715600371360779, "confidence": 0.8432617}, "left_eye_joint": {"x": 0.7510133981704712, "y": 0.6067191362380981, "confidence": 0.91064453}, "head_joint": {"x": 0.7394132614135742, "y": 0.6004706621170044, "confidence": 0.9301758}, "right_eye_joint": {"x": 0.7329676747322083, "y": 0.6067665815353394, "confidence": 0.8515625}, "right_ear_joint": {"x": 0.7206884026527405, "y": 0.5973488092422485, "confidence": 0.77197266}, "left_shoulder_1_joint": {"x": 0.7764022350311279, "y": 0.5364811420440674, "confidence": 0.8930664}},
        "frameNumber":  0,
        "timestamp": 0.3,
    }],
}

# Make the POST request
response = requests.post(url, json=data)

# Check the response status code
if response.status_code == 200:
    print("Request was successful!")
    print("Response Data:", response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Response Content:", response.content) 
