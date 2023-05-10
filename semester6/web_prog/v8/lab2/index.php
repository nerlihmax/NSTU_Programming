<?php

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');

session_start();

$books = pg_query(
  $db, 'SELECT teachers.id,
    position.name as position,
    degree.name as degree,
    courses.name as courses,
    teachers.surname as teacher,
    teachers.room_number
    from teachers as teachers
    inner join position on position.id = teachers.position
    inner join degree on degree.id = teachers.degree
    inner join courses on courses.id = teachers.courses 
    where ' . (isset($_GET['position']) ? "lower(position.name) like lower('%" . pg_escape_string($_GET['position']) . "%')" : "1=1")
);

if (isset($_POST["log_out"])) {
  session_destroy();
  header("location:index.php");
}

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (isset($_POST["log_out"])) {
  session_destroy();
  header("location:index.php");
}

?>

<html>
<style>
  .status {
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 4px 0;
  }
</style>

<head>
  <title>Web - lab2</title>
  <link rel="stylesheet" type="text/css" href="../style.css">
</head>


<body>
  <header class="header">
    <h1 class="h1">Лабораторная работа №2</h1>
    <button onclick="window.location.href = 'http://217.71.129.139:5473/lab2/sources.html'" class="dump">Исходный
      код</button>

  </header>
  <main class="body">
    <div>
      <div class="status">
        <span>
          <?= isset($_SESSION['login']) ? 'Пользователь: ' . $_SESSION['login'] . '.' : 'Вход не выполнен! ' ?>
        </span>
        <?php
        if ($authenticated) {
          echo '<form method="post" action="index.php"><p><button type="submit" name="log_out">Выход</button></p></form>';
        } else {
          echo '<form method="get" action="auth.php"><p><button type="submit">Вход</button></p></form>';
        }
        ?>
      </div>

      <h2>Поиск по должности: </h2>
      <form action="index.php" method="get">
        <div>
          <input name="position" id="position" placeholder="Введите должность" />
          <button type="submit">Поиск</button>
        </div>
      </form>
      <br>
      <?php
      if ($authenticated && $_SESSION['group'] > 1) {
        echo '<a href="insert.php">Добавить запись</a>';
      }
      ?>
      <table>
        <tr>
          <th>ID</th>
          <th>Должность</th>
          <th>Учёная степень</th>
          <th>Курс</th>
          <th>Фамилия</th>
          <th>Аудитория</th>
          <?php
          if ($authenticated && $_SESSION['group'] >= 1) {
            echo "<th></th>";
          }
          if ($authenticated && $_SESSION['group'] >= 2) {
            echo "<th></th>";
          }
          ?>
        </tr>
        <?php
        while ($line = pg_fetch_array($books, null, PGSQL_ASSOC)) {
          echo "\t<tr>\n";
          foreach ($line as $col_value) {
            ?>
            <td>
              <?= $col_value ?>
            </td>
            <?php
          }
          if ($authenticated && $_SESSION['group'] >= 1) {
            echo "\t<td><a href=\"update.php?id=" . $line["id"] . "\">Обновить</a></td>";
            if ($authenticated && $_SESSION['group'] >= 2) {
              echo "\t<td><a href=\"delete.php?id=" . $line["id"] . "\">Удалить</a></td>";
            }
          }
        }
        ?>
        </tr>
      </table>
    </div>
  </main>
</body>

</html>