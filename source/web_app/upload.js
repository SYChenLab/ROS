/* 

upload.js 

upload csc file

ML Platform

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
*/
document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file', document.getElementById('customFile').files[0]);

    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
        headers: {
            'Accept': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('responseMessage').textContent = data.message;
        })
        .catch(error => {
            document.getElementById('responseMessage').textContent = 'error:' + error.message;
        });
});