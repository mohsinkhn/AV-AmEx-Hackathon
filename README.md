## AV-AmEx-Hackathon

Code for 2nd place solution (private LB) for American Express hackathon hosted by Analytics Vidhya. To reproduce results for LB, make approriate changes to `config.py` and run `bash run_all.sh`. Also, make sure all libraries have versions mentioned below.

### Dependencies:
  * numpy >= 1.14.2
  * pandas >= 0.23.4
  * scipy >= 1.1.0
  * sklearn >= 0.20.0
  * **numba >= 0.36.2**
  * **lightgbm >= 2.1.2**
  
  
  ### Approach
  
  #### Validation: 
  Initially, I decided on two validation sets - last day and last 2 days. Since, most of my features were giving improvement in both and validation sets and improvements were reflected in public leaderboard, I decided to stick with last day's data as valiadtion.
  
  #### Features:
  Following features were used:
   * **Time interval features** - Time difference w.r.t last Ad shown and next Ad shown. Time differences for specific product/webpage_id as well.
   * **Expanding count features** - How many times particular ads has been shown so far. Also, for products and webpage_id for a given user, expading counts were used. Hypothesis is that based on how many times a ceratin Ad is shown, its click ratio might vary between user groups
   * **Historical Click features** - How many times an user has clicked Ads in past (also, for given product and on given webpage)
   * **Overall Counts** - Based on how an Ad is shown overall, its click ratio might change. Combination count were also added to make it easier for model to learn deeper interactions
   * **Likelihood Encoding** - Likelihood encoding for different combination to make it easier for model to learn deeper interactions
   * **Ratio features** - Since, its hard for decision trees to learn ratio's. Some ratio were explicitly added.
   * **User-User similarity features** - To help model pack similar users together, user ad counts on for different products, wepbpage_ids and day of week were used.
   * **Historical Log features** - Since, historical logs had information on what Ads a user has clicked in past. Total counts, product counts, dayofweek counts etc. were  calculated. Also, statistics on how many ads were shown to user at the same time were used.
   
   Finally for test predictions, I had decided to average 3 runs with different seeds to help generalize better (but forgot to change seed in loop :'( )
   
   For making predictions on test set, two training sets were used
   
     * Complete training data
     * Training data leaving out first date
   and predictions were averaged
   

