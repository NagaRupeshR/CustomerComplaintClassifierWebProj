async function postData(url, data) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const responseData = await response.json();
    console.log('Success:', responseData);
    return responseData;
  } catch (error) {
    console.error('Error during POST request:', error);
  }
}
(async () => {

console.log(await postData("http://127.0.0.1:5001/mlmodel/setModelLevelTrace",{
  "mlt":{
    "model_version_id":"base_version",
    "accuracy":8688888888888889,
    "macro_f1":0.8689826148032109,
    "weighted_f1":0.8689826148032109
  }
}))
console.log(await postData("http://127.0.0.1:5001/mlmodel/setClassLevelTrace",{
  "clt": [
    {
    "model_version_id": "base_version",
    "class_name": "credit_card",
    "recall": 0.815,
    "f1": 0.817,
    "support": 600
  }],
  "model_version_id":"base_version"
}))
console.log(await postData("http://127.0.0.1:5001/mlmodel/setErrorLevelTrace",{
  "elt":{"ClassificationCount":[[489,26,26,17,42,0],
  [32,487,63,17,1,0],
  [17,43,498,35,7,0],
  [21,27,21,523,8,0],
  [38,8,9,12,533,0],
  [0,0,0,0,2,598]],
  "model_version_id":"base_version"
}
}))
console.log(await postData("http://127.0.0.1:5001/mlmodel/setVersionMetrics",{
  "version_summary":{
    "model_version_id":"base_version",
    "accuracy":8688888888888889,
    "worst_class_f1":0.8170426065162907,
    "worst_class_name":"credit_card",
    "misclassification_rate":3128
  }
}))

let get_mv=await JSON.parse(postData("http://127.0.0.1:5001/mlmodel/getModelVersion"))
let get_mlt=await JSON.parse(postData("http://127.0.0.1:5001/mlmodel/getModelLevelTrace",{"model_version_id":"base_version"}))
let get_clt=await JSON.parse(postData("http://127.0.0.1:5001/mlmodel/getClassLevelTrace",{"model_version_id":"base_version"}))
let get_elt=await JSON.parse(postData("http://127.0.0.1:5001/mlmodel/getErrorLevelTrace",{"model_version_id":"base_version"}))
let get_vm=await JSON.parse(postData("http://127.0.0.1:5001/mlmodel/getVersionMetrics"))
// const myData = {"key":"value"};
// const apiEndpoint = 'https://api.example.com/data';
// postData(apiEndpoint, myData);
console.log(get_mv)
console.log(get_mlt)
console.log(get_clt)
console.log(get_elt)
console.log(get_vm)
})();
// var accuracy = {
//     x:[],
//     y:[],
//     type:"scatter"
// };