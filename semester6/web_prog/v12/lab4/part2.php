<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v12');

$query = 'SELECT distinct 
        workers.id,
        workers.name
    from workers
        join documents on workers.id = documents.worker
    where (select count(*) from documents where worker = id) > 0
    order by id';

$result = pg_query($db, $query);

$workers = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($workers, $line);
}
?>
<html>

<head>
    <meta charset="UTF-8">
    <title>Часть 2</title>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h1>Вариант 12</h1>
    <h2>Часть 2</h2>

    <h3>Выберите исполнителя</h3>
    <form class="form">
        <select class="workers" name="reader">
            <?php
            foreach ($workers as $value) {
                echo "<option value=\"" . $value['id'] . "\">" . $value['name'] . "</option>";
            }
            ?>
        </select>
    </form>

    <br>

    <h3>Документы исполнителя</h3>

    <table class="list" border="1">
        <tr>
            <th>Id</th>
            <th>Исполнитель</th>
            <th>Название документа</th>
            <th>Дата выдачи документа</th>
            <th>Дата возврата документа</th>
        <tr>
    </table>

    <script>
        const getDocumentsByWorkerId = async (workerId) => {
            const response = await fetch(`http://217.71.129.139:5444/lab4/documents_by_worker.php?worker=` + workerId);
            const json = await response.json();
            return json;
        };

        const renderList = ({ documents }) => {
            const list = document.querySelector('.list');

            [...list.children].forEach((child, idx) => idx > 0 && child.remove());

            documents
                .map((item) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${item.id}</td> 
                    <td>${item.worker}</td> 
                    <td>${item.document}</td> 
                    <td>${item.date_of_apply}</td> 
                    <td>${item.date_of_return}</td`;
                    return row;
                })
                .forEach((row) => list.appendChild(row));
        };

        const select = document.querySelector('.workers');

        const selectListener = async () => {
            const worker = select.value;
            const documents = await getDocumentsByWorkerId(worker);
            renderList({ documents });
        };

        selectListener();
        select.addEventListener('input', selectListener);
    </script>

</body>

</html>