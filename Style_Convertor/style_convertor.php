<HTML>
<HEAD>
<TITLE>Style Convertor</TITLE>
</HEAD>
<BODY>
<?php
if($_FILES['file']){
  $check = move_uploaded_file($_FILES['file']['tmp_name'], '/var/www/html/img/input.jpg');
  if(!$check){
    echo "アップロードできませんでした";
  }
  $str="./shell.sh ".$_POST['data'];
  exec($str,$out,$rtn);
  if($rtn != 0){
    echo "変換に失敗しました";
  }
}
?>

<form action="./style_convertor.php" method="POST" enctype="multipart/form-data">
  <input type="file" name="file">
  <select name="data">
   <option value="1">composition</option>
   <option value="2">seurat</option>
   <option value="3">gogh</option>
  </select>
  <input type="submit" value="ファイルをアップロードする">
</form>
<img src="img/input.jpg">
<img src="img/output.jpg">
</BODY>
</HTML>
