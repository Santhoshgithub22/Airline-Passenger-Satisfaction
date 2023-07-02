from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


app=Flask(__name__)



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Customer_Type=request.form.get("Customer_Type"),
            Age = float(request.form.get('Age')),
            Type_of_Travel = request.form.get('Type_of_Travel'),
            Class = request.form.get('Class'),
            Flight_Distance = float(request.form.get('Flight_Distance')),
            Inflight_wifi_service = float(request.form.get('Inflight_wifi_service')),
            Departure_or_Arrival_time_convenient = float(request.form.get('Departure_or_Arrival_time_convenient')),
            Ease_of_Online_booking= float(request.form.get('Ease_of_Online_booking')),
            Gate_location = float(request.form.get('Gate_location')),
            Food_and_drink = float(request.form.get('Food_and_drink')),
            Online_boarding = float(request.form.get('Online_boarding')),
            Seat_comfort = float(request.form.get('Seat_comfort')),
            Inflight_entertainment= float(request.form.get('Inflight_entertainment')),
            On_board_service = float(request.form.get('On_board_service')),
            Leg_room_service = float(request.form.get('Leg_room_service')),
            Baggage_handling = float(request.form.get('Baggage_handling')),
            Checkin_service = float(request.form.get('Checkin_service')),
            Inflight_service= float(request.form.get('Inflight_service')),
            Cleanliness= float(request.form.get('Cleanliness')),
            Departure_Delay_in_Minutes = float(request.form.get('Departure_Delay_in_Minutes')),
            Arrival_Delay_in_Minutes = float(request.form.get('Arrival_Delay_in_Minutes')),
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)


        if pred==0:
            results = 'Not Satisfied'
        else:
            results ='Satisfied'

        return render_template('results.html',final_result=results)

if __name__=="__main__":
    app.run(host='127.0.0.1',  port=5001)