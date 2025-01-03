/* 

tasks.js 

show tasks dashboard

ML Platform

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
*/
const statusDict = {
    "STARTED": "PROGRESS",
};

$(document).ready(function () {
    get_tasks();
    setInterval(get_tasks, 5000);
});
function get_tasks() {
    $.ajax({
        url: 'http://localhost:5000/tasks',
        type: 'GET',
        success: function (data) {
            if (data.length === 0) {
                $('#task-card-body').hide();
            } else {
                $('#task-card-body').show();
                $('#task-list').empty();
                data.forEach(function (task) {
                    let taskStatusHtml = "";
                    task.status = statusDict[task.status] || task.status;
                    if (task.status === "SUCCESS" || task.status === "FAILURE") {
                        taskStatusHtml = `
                        <li class="list-group-item d-flex justify-content-between align-items-center task-item">
                        <span class="badge rounded-pill" style="color: black;">${task.task_name}</span>
                        <span class="badge rounded-pill" style="color: black;">${task.task_id}</span>
                        <span class="badge rounded-pill" style="color: black;">${task.result.result_id}</span>
                        <span class="badge rounded-pill" style="color: black;">${task.task_time}</span>
                        <span class="badge rounded-pill bg-${task.color} status-center">
                        <a href="./result.html?param=${task.result.result_id}" target="_blank" style="text-decoration: none; color: inherit;">
                        ${task.status}
                        </a>
                        </span>
                        </li>`;
                    }
                    else {
                        taskStatusHtml = `
                        <li class="list-group-item d-flex justify-content-between align-items-center task-item">
                        <span class="badge rounded-pill" style="color: black;">${task.task_name}</span>
                        <span class="badge rounded-pill" style="color: black;">${task.task_id}</span>
                        <span class="badge rounded-pill" style="color: black;"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;not received yet &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;</span>
                        <span class="badge rounded-pill" style="color: black;">${task.task_time}</span>
                        <span class="badge rounded-pill bg-${task.color} status-center">${task.status}</span>
                        </li>`;
                    }
                    $('#task-list').append(taskStatusHtml);
                });
            }
        },
        error: function (error) {
            console.log('Error fetching task statuses:', error);
        }
    });
}