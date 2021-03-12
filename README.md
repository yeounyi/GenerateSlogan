# Generate Slogans with Phonetic Similarity 🎶
Please note this is a beta version. The output may not be the best slogan ever. And It may take a couple of minutes. <br>
Deeper explanation about this model can be found in <a href='https://yeounyi.github.io/2021/02/23/model.html' target="_blank">my blog</a>.
* <b>\<name\>, \<loc\>, \<year\> </b>are special tokens. You can substitute these tokens with your brand name, brand location or founding year.

#### Examples
* Phonetically similar words with the keyword and the keyword itself is bolded.


```python3 generate_slogan.py -keyword best```

1. the **best buys**.
2. \<name\>. **buys** the **best**.
3. your **best bet**.
4. the **best besee**.
5. \<loc\> ’s **best bets**.


<br>

```python3 generate_slogan.py -keyword cake```

1. **cut** your **cake** in healthy treats.
2. favorite **cake**. your favorite **cookie**.
3. \<name\>. eat your **cake cut**.
4. the best ice **coat** cream **cake** ever tasted.
5. the **cake** is always the **kicker**.

