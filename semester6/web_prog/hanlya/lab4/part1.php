<?php

define('LOG_PATH', __DIR__ . '/log.txt');

function write_log($comment)
{
  $line = [
    'comment' => $comment,
  ];
  $json = json_encode($line);
  file_put_contents(LOG_PATH, "$json\n", FILE_APPEND);
}

function read_log()
{
  $log = [];
  $json_lines = file_get_contents(LOG_PATH);

  foreach (explode("\n", $json_lines) as $line) {
    if (empty($line))
      continue;
    $log[] = json_decode($line, true);
  }

  return $log;
}

$secs_in_year = time() + 60 * 60 * 24 * 365;

$comment = trim($_POST['comment']);

if (!empty($comment)) {
  write_log($comment);
  header('location:part1.php');
}

?>

<html>

<head>
  <meta charset="UTF-8" />
  <title>Часть 1</title>
  <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-validate/1.19.0/jquery.validate.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="flex flex-col mx-auto h-full w-full justify-top items-center py-4 space-y-10 bg-slate-600">

  <div class="bg-gray-100 rounded-xl text-center px-6 py-4 flex flex-col justify-center my-10 items-center space-y-6">
    <h3 class="mt-2">Валидация средствами JavaScript</h3>
    <form id="myForm" method="post" action="part1.php">
      <p>
        <!-- Input -->
      <div class="w-72">
        <div class="relative h-10 w-full min-w-[200px]">
          <input name="comment" id="comment"
            class="peer h-full w-full rounded-[7px] border border-blue-gray-200 border-t-transparent bg-transparent px-3 py-2.5 font-sans text-sm font-normal text-blue-gray-700 outline outline-0 transition-all placeholder-shown:border placeholder-shown:border-blue-gray-200 placeholder-shown:border-t-blue-gray-200 focus:border-2 focus:border-pink-500 focus:border-t-transparent focus:outline-0 disabled:border-0 disabled:bg-blue-gray-50"
            placeholder=" " />
          <label
            class="before:content[' '] after:content[' '] pointer-events-none absolute left-0 -top-1.5 flex h-full w-full select-none text-[11px] font-normal leading-tight text-blue-gray-400 transition-all before:pointer-events-none before:mt-[6.5px] before:mr-1 before:box-border before:block before:h-1.5 before:w-2.5 before:rounded-tl-md before:border-t before:border-l before:border-blue-gray-200 before:transition-all after:pointer-events-none after:mt-[6.5px] after:ml-1 after:box-border after:block after:h-1.5 after:w-2.5 after:flex-grow after:rounded-tr-md after:border-t after:border-r after:border-blue-gray-200 after:transition-all peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[3.75] peer-placeholder-shown:text-blue-gray-500 peer-placeholder-shown:before:border-transparent peer-placeholder-shown:after:border-transparent peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-pink-500 peer-focus:before:border-t-2 peer-focus:before:border-l-2 peer-focus:before:border-pink-500 peer-focus:after:border-t-2 peer-focus:after:border-r-2 peer-focus:after:border-pink-500 peer-disabled:text-transparent peer-disabled:before:border-transparent peer-disabled:after:border-transparent peer-disabled:peer-placeholder-shown:text-blue-gray-500">
            Комментарий
          </label>
        </div>
      </div>

      </p>
      <p><button type="submit" class="px-4 py-2 mt-4 rounded bg-amber-400">Отправить</button></p>
    </form>

    <script>
      $("#myForm").on("submit", (event) => {
        const isCommentValid = $("#comment").val().length > 0
        if (!isCommentValid) {
          event.preventDefault()
          let res = ""
          if (!isCommentValid) res = "Комментарий не может быть пустым. "
          alert(res)
        }
      })
    </script>
  </div>


  <div class="bg-gray-100 rounded-xl text-center px-6 py-4 flex flex-col justify-center my-10 items-center space-y-6">
    <h3 class="mt-2">Валидация средствами Jquery</h3>
    <form id="myForm2" method="post" action="part1.php">
      <p>
        <!-- Input -->
      <div class="w-72">
        <div class="relative h-10 w-full min-w-[200px]">
          <input name="comment" id="comment"
            class="peer h-full w-full rounded-[7px] border border-blue-gray-200 border-t-transparent bg-transparent px-3 py-2.5 font-sans text-sm font-normal text-blue-gray-700 outline outline-0 transition-all placeholder-shown:border placeholder-shown:border-blue-gray-200 placeholder-shown:border-t-blue-gray-200 focus:border-2 focus:border-pink-500 focus:border-t-transparent focus:outline-0 disabled:border-0 disabled:bg-blue-gray-50"
            placeholder=" " />
          <label
            class="before:content[' '] after:content[' '] pointer-events-none absolute left-0 -top-1.5 flex h-full w-full select-none text-[11px] font-normal leading-tight text-blue-gray-400 transition-all before:pointer-events-none before:mt-[6.5px] before:mr-1 before:box-border before:block before:h-1.5 before:w-2.5 before:rounded-tl-md before:border-t before:border-l before:border-blue-gray-200 before:transition-all after:pointer-events-none after:mt-[6.5px] after:ml-1 after:box-border after:block after:h-1.5 after:w-2.5 after:flex-grow after:rounded-tr-md after:border-t after:border-r after:border-blue-gray-200 after:transition-all peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[3.75] peer-placeholder-shown:text-blue-gray-500 peer-placeholder-shown:before:border-transparent peer-placeholder-shown:after:border-transparent peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-pink-500 peer-focus:before:border-t-2 peer-focus:before:border-l-2 peer-focus:before:border-pink-500 peer-focus:after:border-t-2 peer-focus:after:border-r-2 peer-focus:after:border-pink-500 peer-disabled:text-transparent peer-disabled:before:border-transparent peer-disabled:after:border-transparent peer-disabled:peer-placeholder-shown:text-blue-gray-500">
            Комментарий
          </label>
        </div>
      </div>

      </p>
      <p><button type="submit" class="px-4 py-2 mt-10 rounded bg-amber-400">Отправить</button></p>
    </form>

    <script>
      $(document).ready(function () {
        $("#myForm2").validate({
          rules: {
            comment: {
              required: true,
            },
          },
          messages: {
            comment: {
              required: "Комментарий не может быть пустым. ",
            },
          },
        })
      })
    </script>
  </div>

  <div class="bg-gray-100 rounded-xl text-center px-6 flex flex-col my-10 justify-center items-center">
    <h1 class="text-2xl pb-8 py-3">Журнал</h1>
    <?php
    $log = read_log();
    if (!empty($log)) {
      foreach ($log as $element) {
        $comment = $element['comment'];
        ?>
        <div class="flex flex-col p-2 m-3 outline-dotted">
          <p>
            Комментарий:
            <?= htmlspecialchars($comment) ?>
          </p>
        </div>
        <?php
      }
    }
    ?>
  </div>

</body>

</html>