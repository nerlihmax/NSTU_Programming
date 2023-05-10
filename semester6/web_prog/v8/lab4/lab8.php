<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');
$query = 'SELECT distinct degree.id, degree.name from degree inner join teachers
as teacher on teacher.degree = degree.id
where (select count(*) from teachers where degree = id) > 0 order by id;';

$result = pg_query($db, $query);

$teachers = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($teachers, $line);
}
?>
<html>

<head>
    <title>Web - lab4</title>
    <link rel="stylesheet" type="text/css" href="../style.css">
</head>

<body>
    <header class="header">
        <h1 class="h1">Лабораторная работа №4</h1>
    </header>
    <div>

        <div>
            <h3>Выберите звание сотрудника</h3>
            <form>
                <select class="degree" name="degree">
                    <?php
                    foreach ($teachers as $value) {
                        echo "<option value=\"" . $value['id'] . "\">" . $value['name'] . "</option>";
                    }
                    ?>
                </select>
            </form>
        </div>

        <br>

        <div>
            <h3>Сотрудники c выбранной степенью</h3>
            <table class="list" border="1">
                <tr>
                    <th>ID</th>
                    <th>Должность</th>
                    <th>Учёная степень</th>
                    <th>Курс</th>
                    <th>Фамилия</th>
                    <th>Аудитория</th>
                </tr>
            </table>
        </div>

        <script>
            const getTeachersByDegree = async (degree) => {
                const response = await fetch(`http://217.71.129.139:5473/lab4/ajax.php?degree=` + degree);
                const json = await response.json();
                console.log(json);
                return json;
            };

            const renderList = ({ teachers }) => {
                const list = document.querySelector('.list');

                [...list.children].forEach((child, idx) => idx > 0 && child.remove());

                teachers
                    .map((teacher) => {
                        const row = document.createElement('tr');
                        row.innerHTML = `<td>${teacher.id}</td> 
                    <td>${teacher.position}</td> 
                    <td>${teacher.degree}</td> 
                    <td>${teacher.courses}</td> 
                    <td>${teacher.teacher}</td> 
                    <td>${teacher.room_number}</td`;
                        return row;
                    })
                    .forEach((row) => list.appendChild(row));
            };

            const select = document.querySelector('.degree');

            const app = async () => {
                const degree = select.value;
                const teachers = await getTeachersByDegree(degree);
                renderList({ teachers });
            };

            app();

            select.addEventListener('input', app);
        </script>
    </div>
</body>

</html>