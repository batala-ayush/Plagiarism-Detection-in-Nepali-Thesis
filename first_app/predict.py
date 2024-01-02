import joblib

def predict_lab(ngramval,fingerval,wordval,tfidfsim):
    model = joblib.load('first_app/saved_model_4features.pkl')
    new_obs = [[ngramval,fingerval,wordval,tfidfsim]]
    label = model.predict(new_obs)
    return label
    