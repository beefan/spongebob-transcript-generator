# spongebob transcript generator

### Project Components

1. **Beautiful soup web scraper** that parses [the fan wiki](https://spongebob.fandom.com/wiki/List_of_transcripts) for episode transcripts. Located in directory `web_scraper/`

2. **PyTorch** code that processes a text file, trains a **RNN**, and saves checkpoints for predictions. Located in directory `rnn/`
  * This folder has one `rnnBob.py` which has all the necessary code
  * And one `spongebob_txt_generation.ipynb` which is a jupyter notebook that does some explaining of the code

4. Python script to predict text based on saved model located at `rnn/predict_from_saved_model.py`
5. 1000 words of predicted text plus an entire predicted script at `rnn/example_predictions/`

### Example Prediction Text
**SpongeBob:** Hey, you can have the greatest thing you go, Mr. Krabs. 

**Squidward:** You don't know. I have the little old little more thing is a few friend, I can't do it! I need the Krusty Krab! 

**Mr. Krabs:** And you were the little idea! Mr. Krabs the animals are we have a bunch and you have you two idea! Mr. Krabs?

**Squidward:** And that's the Krusty Krab. I got you two one more idea! 

**SpongeBob:** Oh, that's not a great idea! 

**SpongeBob:** Hey, hey, hey, hey, that's the Krusty Krab. 

**Squidward:** You don't know. 

**SpongeBob:** And this isn't the little animals is not to do it. I got you were to get to my old new king and you can have a great of the king is you can do this. 

### Project Notes
* PyTorch was significantly easier to use than I imagined. I spent a lot of time learning about recurrent neural networks and still trying to grasp the math but when it came down to it, pytorch made the hard parts easy.
* Training was significantly slower than I imagined on my desktop. I use a budget machine with no standalone graphics card. Let the model train for over 24 hours and got some interesting results, but only made it 1/10 through the total training loop. Easy to see how machines with graphics cards dedicated for this might greatly improve the experience of a NLP dev. Probably not going to let the model train for 10 days.