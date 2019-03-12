This project is called Style Convertor. The goal of this project is learning cloud
system and deploying a cloud application. The application is used Machine
Learning and runs on the cloud server. The application can convert the style
of images uploaded to the cloud server. The web page
is implemented in HTML and PHP. First, you have to select an image you want
to convert. Second, you select a style from the pulldown. Finally, you push the
upload button to display the converted image. The shell script is called from
PHP and selects appropriate commands executing machine learning program.
The machine learning program is implemented in Python and converts the style.
I used the program placed in https://github.com/yusuketomoto/chainer-fast-neuralstyle.
The program is made with reference to "Perceptual Losses for Real-Time Style Transfer and Super Resolution" [Johnson, 2016].
