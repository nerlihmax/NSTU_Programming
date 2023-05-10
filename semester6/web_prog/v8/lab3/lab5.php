<?php
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');

$teachers = pg_query(
  $db,
  'SELECT teachers.id,
    position.name as position,
    degree.name as degree,
    courses.name as courses,
    teachers.surname as teacher,
    teachers.room_number
    from teachers as teachers
    inner join position on position.id = teachers.position
    inner join degree on degree.id = teachers.degree
    inner join courses on courses.id = teachers.courses;'
);
?>

<html>
<style>
  #v-legend {
    display: inline-block;
    transform: rotateZ(-90deg);
    position: absolute;
    top: 50%;
    left: -20%;
  }

  .graph {
    display: flex;
    gap: 2px;
    align-items: center;
    justify-content: center;
  }

  .h3 {
    text-align: center;
  }

  #img {
    border-radius: 20px;
  }

  .rotate {
    transform: rotateZ(-90deg) translateY(80px);
  }
</style>

<head>
  <title>Web - lab3 - Part 1</title>
  <link rel="stylesheet" type="text/css" href="../style.css">
</head>

<body>
  <div id="app">
    <header class="header">
      <h1 class="h1">Лабораторная работа №3</h1>
    </header>

    <main class="main">
      <table class="table">
        <tr>
          <th>ID</th>
          <th>Должность</th>
          <th>Учёная степень</th>
          <th>Курс</th>
          <th>Фамилия</th>
          <th>Аудитория</th>
        </tr>
        <?php
        while ($line = pg_fetch_array($teachers, null, PGSQL_ASSOC)) {
          echo "\t<tr>\n";
          foreach ($line as $col_value) {
            ?>
            <td>
              <?= $col_value ?>
            </td>
            <?php
          }
        }
        ?>
        </tr>
      </table>

      <div class="graph">
        <h3 class="rotate">Количество сотрудников</h3>
        <img id="image" src="graphic.php" alt="Сотрудники">
      </div>
      <h3 class="h3">Учёная степень</h3>

    </main>
  </div>
</body>

</html>