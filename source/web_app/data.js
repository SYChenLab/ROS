/* 

data.js 

data display  rename and delete 

ML Platform

Academia Sinica IBMS SYC`LAB
whuang022@gmail.com
*/

$(document).ready(function () {
    // Load file list when page loads
    loadFileList();

    // Load file list using AJAX
    function loadFileList() {
        $.ajax({
            url: 'http://localhost:5000/get_csv_files', // list csv files
            method: 'GET',
            success: function (data) {
                $('#file-list').empty();  // Clear existing list
                data.files.forEach(function (file) {
                    const listItem = `
                            <a href="#" class="list-group-item list-group-item-action">
                                <strong>${file.name}</strong><br>
                                Last modified: ${file.modified_time}<br>
                                <br>
                                 <br>
                                <div class="float-right">
                                    <button class="btn btn-secondary btn-sm ml-2 rename-btn" data-filename="${file.name}">Rename</button>
                                    <button class="btn btn-danger btn-sm delete-btn" data-filename="${file.name}">Delete</button>
                                </div>
                            </a>
                                `;
                    $('#file-list').append(listItem);
                });
                // bind delete btn to deleteFile()
                $('.delete-btn').click(function () {
                    const filename = $(this).data('filename');
                    deleteFile(filename);
                });
                // bind rename btn to rename file
                $('.rename-btn').click(function () {
                    const filename = $(this).data('filename');
                    $('#renameModal').data('filename', filename).modal('show');
                });
            }
        });
    }

    // delete_file onclick
    function deleteFile(filename) {
        if (confirm(`conform to delete ${filename}?`)) {
            $.ajax({
                url: 'http://localhost:5000/delete_file',
                method: 'POST',
                data: { filename: filename },
                success: function () {
                    loadFileList();
                }
            });
        }
    }

    // rename file onclick
    $('#rename-file').click(function () {
        const old_file_name = $('#renameModal').data('filename');
        const new_file_name = $('#new-filename').val();
        if (new_file_name) {
            $.ajax({
                url: 'http://localhost:5000/rename_file',
                method: 'POST',
                data: { old_filename: old_file_name, new_filename: new_file_name },
                success: function () {
                    $('#renameModal').modal('hide');
                    loadFileList();
                }
            });
        }
    });
});
