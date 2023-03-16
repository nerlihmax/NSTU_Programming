<html>

<head>
    <title>Лабораторная №3</title>
    <meta charset="utf-8">
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }

        #ckeditor {
            width: 75%;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №3</h1>
    <h1>Часть 2</h1>
    <h3>Вариант 15</h3>

    <?php
    include_once("fckeditor/fckeditor.php");
    session_start();
    ?>

    <h3>Комментарий</h3>
    <form action="journal.php" method="post" target="_self">
        <div id="ckeditor">
            <?php
            $oFCKeditor = new FCKeditor('text_input');
            $oFCKeditor->BasePath = '/lab3/fckeditor/';
            $oFCKeditor->Value = '';
            $oFCKeditor->Create();
            ?>
        </div>
        <br>
        <h3><span>Имя комментатора: </span><br><input type="text" required name="username"></h3>
        <br>
        <img src="/lab3/kcaptcha/index.php?<?php echo session_name() ?>=<?php echo uniqid() ?>"></p>
        <p><span>Код подтверждения: </span><br><input type="text" required name="code"></p>
        <input type="submit" value="Submit">
    </form>
    </div>
</body>

</html>