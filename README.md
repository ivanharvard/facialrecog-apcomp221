If you want to run this entire script on your device you could either:

A. Copy each individual file on your own system and run it.
B. Message Ivan on Whatsapp (or call if it's urgent) to push to github, and then
   clone everything you need and run it on your own system then.
C. Open the Live Share link and use the test video file instead of your camera.

Run the test.py file if you just care about first-time setup for your camera. 
You should see two windows: a blurred window and a regular feed. Press q to exit.
If it doesn't work, make sure you're not running on WSL, and that you have 
installed opencv-python. And then make sure that your default camera is activated.

Alternatively, run the test.py file with a test video file. If you wish to submit your own video, for the sake of storage (and because LiveShare doesn't like large files), please upload it to [8mb.video](https://8mb.video/) before adding it to the test_videos/ directory. Select whatever quality is appropriate.

When finished, we'll run everything through the main file.

For the faces folder, filenames should be the names of the people shown in the photo of the form "John_Doe.jpg" or ".png".

If installing face_recognition fails, run:

sudo apt update
sudo apt install build-essential cmake libboost-all-dev python3-dev