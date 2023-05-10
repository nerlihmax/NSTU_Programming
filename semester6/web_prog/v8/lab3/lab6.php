<html>
<style>
  .form {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .form__flags {
    display: flex;
    gap: 4px;
    align-items: center;
  }

  .form__flags__div {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .ckeditor {
    padding: 8px;
  }

  .img_cap {
    max-width: 300px;
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

    <?php
    include_once("fckeditor/fckeditor.php");
    session_start();
    ?>

    <main class="main">
      <form action="journal.php" method="post" target="_self" class="form">
        <div class="form__flags">
          <p>Оформления текста:</p>
          <div class="form__flags">
            <p>Оформления текста:</p>
            <div class="form__flags__div">
              <p><input TYPE="checkbox" id="bold" value="bold" name="styles[]"></p>
              <label for="cat">Жирный</label>
            </div>
            <div class="form__flags__div">
              <p><input TYPE="checkbox" id="italic" value="italic" name="styles[]"></p>
              <label for="dog">Курсив</label>
            </div>
            <div class="form__flags__div">
              <p><input TYPE="checkbox" id="dotted" value="dotted" name="styles[]"></p>
              <label for="turtle">В горошек</label>
            </div>
          </div>
        </div>

        <p>Комментарий: </p>
        <div id="ckeditor" class="ckeditor">
          <?php
          $oFCKeditor = new FCKeditor('comment');
          $oFCKeditor->BasePath = 'fckeditor/';
          $oFCKeditor->Value = '';
          $oFCKeditor->Create();
          ?>
        </div>
        <div>
          <label for="username">Имя: </label>
          <input name="username" required placeholder="Имя комментатора" />
        </div>
        <img class="img_cap" src="kcaptcha/index.php?<?php echo session_name() ?>=<?php echo session_id() ?>">
        </p>
        <div>
          <label for="code">Код подтверждения: </label>
          <input name="code" required placeholder=" " />
        </div>
        <button type="submit">Отправить</button>
      </form>
    </main>
  </div>
</body>

</html>