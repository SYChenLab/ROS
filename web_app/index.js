/* 

index.js 

ML task setting and submit

ML Platform

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
*/
var use_encode = 'True'
// add switch event
const switchElement = document.getElementById('mySwitch');
switchElement.addEventListener('change', function () {
    if (switchElement.checked) {
        use_encode = 'True';
    } else {
        use_encode = 'False';
    }
});
// show spinner
function showLoadingSpinner() {
    const spinnerHTML = `
    <div class="spinner-border" role="status">
        <span class="visually-hidden">Loading...</span>
    </div>
`;
    document.getElementById('feature-selection').innerHTML = spinnerHTML;
    document.getElementById('target-selection').innerHTML = spinnerHTML;
}
// hide spinner
function hideLoadingSpinner() {
    document.querySelectorAll('.spinner-border').forEach(spinner => spinner.remove());
}
// update Features by selected csv file
function updateFeatures(selectElement) {
    var selectedValue = selectElement.value;
    const input_index_col = document.getElementById("input_index_col").value;
    //
    document.getElementById('feature-selection-container').style.display = 'block';
    document.getElementById('target-selection-container').style.display = 'block';
    showLoadingSpinner();
    const formData = new FormData();
    formData.append('file', selectedValue);
    formData.append('input_index_col', input_index_col);

    $.ajax({
        url: 'http://localhost:5000/get_columns',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            const columns = response.columns;

            let targetOptions = '';
            columns.forEach(function (column) {
                targetOptions += `<span class="badge rounded-pill bg-primary" style="cursor:pointer" onclick="toggleTargetSelection(this)">${column}</span> `;
            });
            document.getElementById('target-selection').innerHTML = targetOptions;

            let featureOptions = '';
            columns.forEach(function (column) {
                featureOptions += `<span class="badge rounded-pill bg-primary" style="cursor:pointer" onclick="toggleSelection(this)">${column}</span> `;
            });
            document.getElementById('feature-selection').innerHTML = featureOptions;
            document.getElementById('feature-selection-container').style.display = 'block';
            document.getElementById('target-selection-container').style.display = 'block';

            hideLoadingSpinner();
        },
        error: function (error) {
            console.log('Error:', error);
            hideLoadingSpinner();
        }
    });
}

// list CSV files on server
$(document).ready(function () {
    $.ajax({
        url: 'http://localhost:5000/get_csv_files',
        method: 'GET',
        success: function (response) {
            const files = response.files;
            let fileOptions = '';
            files.forEach(function (file) {
                fileOptions += `<option value="${file.name}">${file.name}</option>`;
            });
            document.getElementById('input_csv').innerHTML += fileOptions;
        },
        error: function (error) {
            console.log('Error fetching CSV files:', error);
        }
    });
});


function toggleSelection(element) {

    const selectedTarget = document.querySelector('.bg-success.target-selected');
    if (selectedTarget && selectedTarget.textContent === element.textContent) {
        alert('This feature is already selected as the target. Please choose another feature.');
        return;
    }

    element.classList.toggle('bg-success');
}

// select all features at once
function selectAllFeatures() {
    document.querySelectorAll('#feature-selection .badge').forEach(function (element) {
        if (!element.classList.contains('bg-success')) {
            element.classList.add('bg-success');
        }
    });
}

function toggleTargetSelection(element) {

    const selectedTarget = document.querySelector('.bg-success.target-selected');
    if (selectedTarget) {
        selectedTarget.classList.remove('bg-success', 'target-selected');
    }

    element.classList.add('bg-success', 'target-selected');

    document.querySelectorAll('#target-selection .badge').forEach(function (badge) {
        if (badge !== element) {
            badge.classList.remove('bg-primary');
            badge.classList.add('bg-secondary');
        }
    });


    element.classList.add('disabled');


    document.querySelectorAll('#feature-selection .badge').forEach(function (badge) {
        if (badge.textContent === element.textContent) {
            badge.classList.remove('bg-success');
        }
    });
}


let availableParameters = [];


function fetchModelParameters(selectElement) {
    const algorithm = selectElement.value;
    const paramsContainer = document.getElementById("algorithm-params-container");

    availableParameters = [];

    const selectedParametersDiv = document.getElementById("selected-parameters");
    selectedParametersDiv.innerHTML = '';

    document.getElementById("parameter-selection-container").style.display = "none";

    if (!algorithm) {
        paramsContainer.style.display = "none";
        return;
    }

    fetch(`http://localhost:5000/get_algorithm_params/${algorithm}`)
        .then(response => response.json())
        .then(data => {
            const params = data.parameters;
            console.log(params)
            const paramsDiv = document.getElementById("algorithm-params");
            paramsDiv.innerHTML = "";
            if (params && params.length > 0) {
                availableParameters = params;

                availableParameters.forEach(param => {
                    const badge = document.createElement("span");
                    badge.classList.add("badge", "rounded-pill", "bg-secondary", "ms-2");
                    badge.textContent = param;
                    badge.style.cursor = "pointer";
                    badge.onclick = () => selectParameter(param);

                    paramsDiv.appendChild(badge);
                });

                paramsContainer.style.display = "block";
            } else {
                paramsContainer.style.display = "none";
            }
        })
        .catch(error => {
            console.error("Error fetching parameters:", error);
            paramsContainer.style.display = "none";
        });
}


function selectParameter(parameterKey) {
    const selectedParametersDiv = document.getElementById("selected-parameters");

    const selectedParam = availableParameters.find(param => param.key === parameterKey);

    const selectedParamElement = document.createElement("div");
    selectedParamElement.classList.add("mb-2", "d-flex", "align-items-center");

    const badge = document.createElement("span");
    badge.classList.add("badge", "rounded-pill", "bg-secondary", "parameter-badge");
    badge.textContent = parameterKey;

    const valueInput = document.createElement("input");
    valueInput.type = "text";
    valueInput.classList.add("form-control", "ms-2");
    valueInput.placeholder = `Enter value for ${parameterKey}`;
    valueInput.setAttribute("data-parameter", parameterKey);

    const removeButton = document.createElement("button");
    removeButton.classList.add("btn", "btn-danger", "btn-sm", "ms-2");
    removeButton.textContent = "Remove";
    removeButton.onclick = () => removeParameter(selectedParamElement, parameterKey);

    selectedParamElement.appendChild(badge);
    selectedParamElement.appendChild(valueInput);
    selectedParamElement.appendChild(removeButton);
    selectedParametersDiv.appendChild(selectedParamElement);
    document.getElementById("parameter-selection-container").style.display = "block";
    removeBadgeFromAlgorithmList(parameterKey);
}

function removeParameter(paramElement, parameterKey) {
    const selectedParametersDiv = document.getElementById("selected-parameters");
    selectedParametersDiv.removeChild(paramElement);
    if (selectedParametersDiv.children.length === 0) {
        document.getElementById("parameter-selection-container").style.display = "none";
    }
    reAddBadgeToAlgorithmList(parameterKey);
}


function removeBadgeFromAlgorithmList(parameterKey) {
    const paramsDiv = document.getElementById("algorithm-params");
    const badges = paramsDiv.querySelectorAll(".badge");

    badges.forEach(badge => {
        if (badge.textContent.trim() === parameterKey) {
            badge.remove();
        }
    });
}

function reAddBadgeToAlgorithmList(parameterKey) {
    const paramsDiv = document.getElementById("algorithm-params");
    const existingBadges = paramsDiv.querySelectorAll(".badge");
    const isAlreadyAdded = Array.from(existingBadges).some(badge => badge.textContent.trim() === parameterKey);
    if (isAlreadyAdded) {
        return;
    }

    const selectedParam = availableParameters.find(param => param === parameterKey);

    if (selectedParam) {
        const badge = document.createElement("span");
        badge.classList.add("badge", "rounded-pill", "bg-secondary", "ms-2");
        badge.textContent = parameterKey;
        badge.style.cursor = "pointer";
        badge.onclick = () => selectParameter(parameterKey);
        paramsDiv.appendChild(badge);
    }
}
//add parameters for algorithm
function addParameter() {
    const key = document.getElementById("parameter_key").value;
    const value = document.getElementById("parameter_value").value;
    if (key && value) {
        let existingParam = document.querySelector(`#algorithm-params div[data-key="${key}"]`);

        if (existingParam) {
            existingParam.querySelector(".badge").textContent = `${key}: ${value}`;
            let removeBtn = existingParam.querySelector(".remove-btn");
            if (!removeBtn) {
                removeBtn = document.createElement("button");
                removeBtn.type = "button";
                removeBtn.classList.add("btn", "btn-danger", "btn-sm", "ms-2", "remove-btn");
                removeBtn.textContent = "Remove";
                removeBtn.onclick = function () {
                    removeParameter(existingParam);
                };
                existingParam.appendChild(removeBtn);
            }
            removeBtn.style.display = "inline-block";
        } else {
            const paramsDiv = document.getElementById("algorithm-params");
            const newParamElement = document.createElement("div");
            newParamElement.classList.add("mb-2");
            newParamElement.setAttribute("data-key", key);
            const badge = document.createElement("span");
            badge.classList.add("badge", "rounded-pill", "bg-secondary");
            badge.textContent = `${key}: ${value}`;
            newParamElement.appendChild(badge);
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.classList.add("btn", "btn-danger", "btn-sm", "ms-2", "remove-btn");
            removeBtn.textContent = "Remove";
            removeBtn.onclick = function () {
                removeParameter(newParamElement);
            };
            newParamElement.appendChild(removeBtn);
            paramsDiv.appendChild(newParamElement);
        }
        document.getElementById("parameter_value").value = "";
    }
}

// show message
function showModal(message) {
    document.getElementById('modalMessage').innerText = message;
    var myModal = new bootstrap.Modal(document.getElementById('messageModal'));
    myModal.show();
}

// submit project
function submitForm() {
    // get input args of ml project
    const projectName = document.getElementById("project_name").value;
    const input_index_col = document.getElementById("input_index_col").value;
    const trainingFile = document.getElementById("input_csv").value;
    const algorithm = document.getElementById("algorithm").value;
    const random_seed = document.getElementById("random_seed").value;
    let algorithmParams = {};
    // get algorithm params
    document.querySelectorAll("#selected-parameters .mb-2").forEach(function (paramDiv) {
        const badge = paramDiv.querySelector(".parameter-badge");
        const valueInput = paramDiv.querySelector("input");
        const paramKey = badge.textContent;
        const paramValue = valueInput.value.trim();
        if (paramValue) {
            algorithmParams[paramKey] = paramValue;
        }
    });
    // get selected features
    const selectedFeatures = [];
    document.querySelectorAll("#feature-selection .badge.bg-success").forEach(function (featureBadge) {
        selectedFeatures.push(featureBadge.textContent.trim());
    });
    // get the selected target
    const selectedTarget = document.querySelector('.bg-success.target-selected')?.textContent.trim();
    // Construct the JSON object
    const formData = {
        project_name: projectName,
        input_csv: trainingFile,
        input_index_col: input_index_col,
        'use_encode': use_encode,
        random_seed: random_seed,
        algorithm: algorithm,
        algorithm_params: algorithmParams,
        features: selectedFeatures,
        target: selectedTarget
    };
    // POST to server
    fetch('http://localhost:5000/submit_project', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data.message);
            showModal(data.message);
            setTimeout("location.href='./tasks.html'", 5000);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while submitting the project.');
        });
}


