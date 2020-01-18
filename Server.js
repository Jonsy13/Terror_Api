const express = require('express');
const bodyParser = require('body-parser');
const cors = require("cors");
const { PythonShell } = require('python-shell');

const app = express();
app.use(bodyParser.json());
app.use(cors());

app.listen(process.env.PORT || 3000, () => {
    console.log(`Server is listening on PORT {process.env.PORT}`);
})


app.get('/', (req, res) => {
    console.log("Home");
})
// app.post('/Predict', (req, res) => {
app.post('/Predict', (req, res) => {


    const Month = req.body.Month;
    const Day = req.body.Day;
    const Ext = req.body.Extended;
    const Lat = req.body.Latitude;
    const Long = req.body.Longtitude;
    const Mul = req.body.Multiple;
    const Suicide = req.body.Suicide;
    const Atype = req.body.Atype;
    const Ttype = req.body.Ttype;
    const Ind = req.body.Individual;
    const Wtype = req.body.Wtype;

    console.log(req.body);


    let options = {
        mode: 'text',
        pythonPath: 'python',
        pythonOptions: ['-u'], // get print results in real-time
        scriptPath: './',
        args: [Month, Day, Ext,Lat,Long,Mul,Suicide,Atype,Ttype,Ind,Wtype]
    };

    PythonShell.run('terror_success_prediction-2.py', options, function (err, results) {
        if (err) throw err;
        // results is an array consisting of messages collected during execution
        console.log('results: %j', results);
        res.json(results);
    });


    console.log("Server connected");
})