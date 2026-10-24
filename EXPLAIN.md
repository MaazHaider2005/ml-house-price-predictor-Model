# Study guide: be ready to explain your code

Your assignment says you must be able to explain every line. Read the code files with this guide next to them, and
try rewriting parts in your own words. Likely questions:

**Training (`train_models.py`)**
1. *Why split into train and test?* To measure the models on data they have never seen. Scoring on training data would flatter them.
2. *What are R² and MAE?* R² = share of the price differences the model explains (1.0 is perfect). MAE = average size of the miss, in dollars.
3. *Why predict the log of the price?* Prices are skewed (a few mansions cost millions). In log form, errors are proportional, and a straight line fits much better. `exp` converts back to dollars.
4. *Why one-hot encode zip code?* A zip code is a label, not a quantity (98103 isn't "bigger" than 98004). One-hot makes one 0/1 column per zip so the model learns a price level per area.
5. *Why log the square footage?* A 500 sq ft extension adds more to a small home than a huge one, which is a percentage effect.
6. *What is a Pipeline?* It chains steps (prepare the data, then fit the model) so the exact same steps run at prediction time.
7. *Why refit on all the data at the end?* The scores come from the held-out test; the saved model then gets to learn from every example.

**Classification (the price tier model)**
8. *What is the difference between regression and classification?* Regression predicts a number (the price in dollars). Classification predicts a category (Affordable, Mid-range or Premium).
9. *How did you make the categories?* The cut-off prices are the one-third and two-thirds points of all sale prices (`np.quantile`), so each tier holds about the same number of houses. They are saved in `metrics.json`.
10. *Why is accuracy a fair score here?* The three tiers are about equal in size, so guessing would be right about 33% of the time. 82.9% is far above that.
11. *Why try three models and use cross-validation?* Cross-validation tests each model on 5 different slices of the training houses, which is more reliable than one split. The best scorer is kept.
12. *Why StandardScaler for Logistic Regression?* It compares the size of numbers, so columns like `sqft_basement` (thousands) and `view` (0 to 4) should be on a similar scale.
13. *What does `predict_proba` return?* The model's chance for each tier. The tier with the highest chance is the prediction, and that chance is the "confidence".

**Backend (`main.py`)**
14. *What does `BaseModel` / `Field` do?* It validates incoming JSON (types and allowed ranges) and returns a clear error if something is off.
15. *Why load the models at the top, not inside the endpoint?* Loading once at startup is fast; loading on every request would be slow.
16. *Why build a one-row DataFrame?* The pipelines were trained on a table with named columns, so they expect the same shape.
17. *Where does the price range come from?* In testing, 8 in 10 houses were within `within_80_percent` of the estimate (saved in `metrics.json`).
18. *What is CORS?* A browser rule about pages calling other addresses; the middleware relaxes it. Not needed when the page is served by the same server.

**Frontend (`static/index.html`)**
19. *How does the page talk to the backend?* `fetch()` sends JSON to the endpoint with POST, then the result is drawn into the page. Both models get the same house, so the page asks them at the same time with `Promise.all`.
20. *How is the dataset connected to the page?* `/options` fills the zip-code list and the accuracy numbers from the CSV and training; `/sample` returns a real row so you can compare the estimate and tier with what the house really sold for. The range notes under each field and the grade list use the values that exist in the CSV columns.
