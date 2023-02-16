<html>

<head>
    <title>Web programming NSTU course</title>
    <meta charset="utf-8">
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h3>Вариант 17</h3>
    <?php
    include_once("fckeditor/fckeditor.php");
    session_start();
    ?>

    <h3>Пожелания по виду отдыха</h3>
    <form action="journal.php" method="post" target="_blank">
        <?php
        $oFCKeditor = new FCKeditor('text_input');
        $oFCKeditor->BasePath = '/lab3/fckeditor/';
        $oFCKeditor->Value = 'Заходит однажды в бар улитка и говорит:<br>
        -Можно виски с колой?<br>
        - Простите, но мы не обслуживаем улиток.<br>
        И бармен вышвырнул ее за дверь.<br>
        Через неделю заходит опять эта улитка и спрашивает:<br>
        -Ну и зачем ты это сделал!?<br>';
        $oFCKeditor->Create();
        ?>
        <br>
        <h3><span>Имя туриста: </span><br><input type="text" name="username"></h3>
        <br>
        <img src="/lab3/kcaptcha/index.php?<?php echo session_name() ?>=<?php echo session_id() ?>"></p>
        <p><span>Код подтверждения: </span><br><input type="text" name="code"></p>
        <input type="submit" value="Submit">
    </form>


</body>


</html>