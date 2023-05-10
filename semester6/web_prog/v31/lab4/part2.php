<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v31');

$query = 'SELECT distinct 
        operation.id,
        operation.name
    from operation
        join technological_map on operation.id = technological_map.operation
    where (select count(*) from technological_map where operation = id) > 0
    order by id';

$result = pg_query($db, $query);

$operations = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($operations, $line);
}
?>
<html>

<head>
    <meta charset="UTF-8">
    <title>Часть 2</title>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h1>Вариант 11</h1>
    <h2>Часть 2</h2>

    <h3>Выберите исполнителя</h3>
    <form class="form">
        <select class="operations" name="operation">
            <?php
            foreach ($operations as $value) {
                echo "<option value=\"" . $value['id'] . "\">" . $value['name'] . "</option>";
            }
            ?>
        </select>
    </form>

    <br>

    <h3>Детали обработанные выбранным способом</h3>

    <table class="list" border="1">
        <tr>
            <th>ID</th>
            <th>Название детали</th>
            <th>Вид операции</th>
            <th>Длительность операции в минутах</th>
        <tr>
    </table>

    <script>
        const fetchDetails = async (operation) => {
            const response = await fetch(`http://217.71.129.139:4030/lab4/details_operation.php?operation=` + operation);
            const json = await response.json();
            return json;
        };

        const renderList = ({ details }) => {
            const list = document.querySelector('.list');

            [...list.children].forEach((child, idx) => idx > 0 && child.remove());

            details
                .map((item) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${item.id}</td> 
                    <td>${item.name}</td> 
                    <td>${item.operation}</td> 
                    <td>${item.duration}</td>`;
                    return row;
                })
                .forEach((row) => list.appendChild(row));
        };

        const select = document.querySelector('.operations');

        const selectListener = async () => {
            const operation = select.value;
            const details = await fetchDetails(operation);
            renderList({ details });
        };

        selectListener();
        select.addEventListener('input', selectListener);
    </script>

</body>

</html>