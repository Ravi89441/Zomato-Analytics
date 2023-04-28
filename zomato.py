# Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# Importing dataset 
zomato_df = pd.read_csv('zomato.csv')

# Lokking first 5 rows of data
print(zomato_df.head())

'''
                                                 url                                            address  ... listed_in(type) listed_in(city)
0  https://www.zomato.com/bangalore/jalsa-banasha...  942, 21st Main Road, 2nd Stage, Banashankari, ...  ...          Buffet    Banashankari
1  https://www.zomato.com/bangalore/spice-elephan...  2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...  ...          Buffet    Banashankari
2  https://www.zomato.com/SanchurroBangalore?cont...  1112, Next to KIMS Medical College, 17th Cross...  ...          Buffet    Banashankari
3  https://www.zomato.com/bangalore/addhuri-udupi...  1st Floor, Annakuteera, 3rd Stage, Banashankar...  ...          Buffet    Banashankari
4  https://www.zomato.com/bangalore/grand-village...  10, 3rd Floor, Lakshmi Associates, Gandhi Baza...  ...          Buffet    Banashankari

'''
# Looking last 5 rows of dataset 
print(zomato_df.tail())

'''
                                                     url                                            address  ... listed_in(type) listed_in(city)
51712  https://www.zomato.com/bangalore/best-brews-fo...  Four Points by Sheraton Bengaluru, 43/3, White...  ...   Pubs and bars      Whitefield
51713  https://www.zomato.com/bangalore/vinod-bar-and...  Number 10, Garudachar Palya, Mahadevapura, Whi...  ...   Pubs and bars      Whitefield
51714  https://www.zomato.com/bangalore/plunge-sherat...  Sheraton Grand Bengaluru Whitefield Hotel & Co...  ...   Pubs and bars      Whitefield
51715  https://www.zomato.com/bangalore/chime-sherato...  Sheraton Grand Bengaluru Whitefield Hotel & Co...  ...   Pubs and bars      Whitefield
51716  https://www.zomato.com/bangalore/the-nest-the-...  ITPL Main Road, KIADB Export Promotion Industr...  ...   Pubs and bars      Whitefield

'''

# Checking Shape of dataframe
print(zomato_df.shape)

'''
(51717, 17)

'''
# Checking features 
print(zomato_df.columns)

'''

Index(['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes',
       'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',
       'approx_cost(for two people)', 'reviews_list', 'menu_item',
       'listed_in(type)', 'listed_in(city)'],
      dtype='object')

'''

# Checking info
print(zomato_df.info())

## Checking statistical summary of all features
print(zomato_df.describe(include= 'all'))

'''
Data columns (total 17 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   url                          51717 non-null  object
 1   address                      51717 non-null  object
 2   name                         51717 non-null  object
 3   online_order                 51717 non-null  object
 4   book_table                   51717 non-null  object
 5   rate                         43942 non-null  object
 6   votes                        51717 non-null  int64
 7   phone                        50509 non-null  object
 8   location                     51696 non-null  object
 9   rest_type                    51490 non-null  object
 10  dish_liked                   23639 non-null  object
 11  cuisines                     51672 non-null  object
 12  approx_cost(for two people)  51371 non-null  object
 13  reviews_list                 51717 non-null  object
 14  menu_item                    51717 non-null  object
 15  listed_in(type)              51717 non-null  object
 16  listed_in(city)              51717 non-null  object
dtypes: int64(1), object(16)

'''
# Checking null values 
print(zomato_df.isnull().sum())

# Droping all unnecessary columns
zomato_df = zomato_df.drop(['url', 'address', 'phone','dish_liked','reviews_list', 'menu_item'], axis=1)
print(zomato_df)

# Checking for duplicates values 
print(zomato_df[zomato_df.duplicated()].columns)

# Dropping duplicates values 
zomato_df = zomato_df.drop_duplicates()
print(zomato_df)

#Analysing "rate" columns since ther are lot of null values present
print(zomato_df['rate'].unique())
print(zomato_df['rate'])


# Defining a function to handel the rating column and also covert them from text to integer form
def treat_ratings(values):
    if (values == 'NEW' or values == '-'):
        return np.nan                                  # NumPy NAN stands for not a number and 
                                                       #np.nan is defined as a substitute for declaring value which are numerical values that are missing values in an array as NumPy is used to deal with arrays in Python
    else :
        values = str(values).split('/')                # since split method splits string into list of int 
                                                       # i.e."4.1/5" will split it into 4.1 and /5 ---> list is =   values=[4.1][/5]
        values = values[0]                             # But here we only need numerator of rating because all the rating are out of 5
                                                       # So we have stated its index value.
        return float(values)                           # Since we want these values in floating form.
        
        
# Now applying the function on "rate" column        
zomato_df.rate = zomato_df['rate'].apply(treat_ratings)
print(zomato_df['rate'].unique()) 
print(zomato_df.info())

# Checking null Values
print(zomato_df.isnull().sum())

# Using fill na method 
print(zomato_df.rate.fillna(zomato_df.rate.mean(), inplace= True))
print(zomato_df.isnull().sum())

# So droping other features null values
print(zomato_df.dropna(inplace = True))
print(zomato_df.isnull().sum())

#Now analysing Restaurant type feature
print(zomato_df['rest_type'].value_counts())
rest_type = zomato_df['rest_type'].value_counts()
print(rest_type)

#so for the better understanding of feature we will groups these rest_type which have less than 1000 counts
other_resto_types = rest_type[rest_type <1000]
print(other_resto_types)

# Cheking types of data
print(type(other_resto_types))

# Defining Function
def handle_rest_type(type):
    if (type in other_resto_types):
        return 'other_resto_types'
    
    else:
        return type
    
zomato_df['rest_type'] = zomato_df['rest_type'].apply(handle_rest_type)
print(zomato_df.head())

print(zomato_df['rest_type'].value_counts())

# Now Analysing where are these restaurants are located i.e. "location" column
print(zomato_df['location'])
print(zomato_df['location'].value_counts())


'''
observation:

1) most of the restaurants are in BTM area i.e. 5056 number of restaurants, since we can infer that this area is highly populated and has good rich in culture.
2) since there are also lot of such location are present in dataset which has less number of restaurants.
3) so it is better for analysis and visualization purpose we can groups these location like we did for rest_type feature.
4) grouping those locations which has less than 500 restaurant counts and storing them in "other lcation" variables.

'''

locations = zomato_df['location'].value_counts() 
other_locations = locations[locations < 500]
print(other_locations)
print(other_locations.value_counts().sum())


# defining function 
def handle_location (location):
    if location in other_locations :
        return 'other locations'
    else:
        return location
    
# applying function
zomato_df['location'] = zomato_df['location'].apply(handle_location)
print(zomato_df['location'].value_counts())

# Now Analysing "Cuisines" feature i.e. Food styles
print(zomato_df['cuisines'])
print(zomato_df['cuisines'].value_counts())


'''
Observation

1) from above cuisines column we say that most of the food items in restaurants are north indan types.
2) then north indian chinese is having 2351 types of items.
3) Also there are lot of such styles of cooking presents which having less in numbers
4) so for good analysis we can groups these cuisines which have less than 100 counts of items in variable "less_num_cuisines"

'''

cuisines = zomato_df['cuisines'].value_counts()
less_num_cuisines = cuisines[cuisines <100]

# definiing function 
def handling_cusines(cuisines):
    if cuisines in less_num_cuisines:
        return 'less_num_cuisines'
    else:
        return cuisines
    
# Applying fun on dataframe
zomato_df['cuisines'] = zomato_df['cuisines'].apply(handling_cusines)
zomato_df['cuisines'].value_counts()
print(zomato_df.head())
print(zomato_df.info())

# Now analysing cost of plate per cuisines in restaurant i.e. � "approx_cost(for two people) feature
print(zomato_df['approx_cost(for two people)'].value_counts())
print(zomato_df['approx_cost(for two people)'].unique())


'''
Observation : 

1) as we see that data type of cost feature is object becoz of comas in between number when approx cost is above 999. i.e. 1,900 so on
2) so we have to remove these comas i.e. ","

'''


# Writing functing to remove "," and convert this column into float
def handling_approx_cost(cost):
    cost = str(cost)
    if "," in cost:
        cost = cost.replace("," , "")           # we simply replaced coma with empty value
        return float(cost)                      # converting data type into float
    else:
        return float(cost)
    
# applying fun on feature
zomato_df['approx_cost(for two people)'] = zomato_df['approx_cost(for two people)'].apply(handling_approx_cost)
print(zomato_df['approx_cost(for two people)'].unique())
print(zomato_df['approx_cost(for two people)'].describe())


'''
Observation :

1) Now we have successfully perform operation on approx_cost feature
2) loweset cost per plate price is Rs. 40 in restaurant
3) highest cost per plate price is 6000 in some restaurants

'''

# Now Analysing listed_in(type) i.e. types of meals.
print(zomato_df['listed_in(type)'])
print(zomato_df['listed_in(type)'].value_counts())


'''
Observations :

1) In restaurants most of the types of meals are for delivery i.e. 25579 i.e most of people tries to order food from restaurants
2) Then 17562 number of meal types are Dine-out i.e. means lot of peoples loves to eat outside from home i.e. in restaurants
3) 3559 number of meals types are in Desserts or sweets
4) 1703 number of types of meals are from cafes
5) 1084 numbers are of Drinks & nightlife data
6) 869 numbers are of buffet type of meals
7) 689 numbers of pubs and bars present in datset of banglore

'''


# Since "location" and "listed_in(city)" both gives same meaning. i.e. both columns shows area or location of restaurants in dataset.
zomato_df = zomato_df.drop(columns= 'listed_in(city)')
print(zomato_df.head())
print(zomato_df.columns)

# Number of restaurant in different locations
print(zomato_df['location'].value_counts())


# Plotting Locations
counts = zomato_df['location'].value_counts().sort_index()
fig = plt.figure(figsize=(20, 7))
ax = fig.gca()
 
counts.plot.bar(ax = ax, color='Orange', edgecolor= 'black' )
ax.set_title( 'location'+ ' wise '+' Restaurants')
ax.set_xlabel('location') 
ax.set_ylabel("counts")
plt.show()

'''



'''


'''
Observation :

1) As BTM place has above 5000 plus number of restaurants
2) That means BTM is place where most of the people tries to go there or order some cuisines.
3) And Other location means lovcation which have less than 500 restaurents in dataset are quite high counts i.e. almost 8000 number of places are there which having <500 restaurants.

'''

# Checking Restaurants way of delivery for cuisines.
print(zomato_df['online_order'].value_counts())
counts = zomato_df['online_order'].value_counts().sort_index()
fig = plt.figure(figsize=(5,6))
ax = fig.gca()
 
counts.plot.bar(ax = ax, color='yellow')
ax.set_title( 'online_order'+' counts')
ax.set_xlabel('online orders') 
ax.set_ylabel("counts")
plt.show()

'''


'''


'''
Observations :

1) In Dataset 30228 restaurents have online delivery facility
2) 20814 restaurants are not having online delivery service.

'''

# Checking wheater restaurants have Booking Table facility or not!
print(zomato_df['book_table'].value_counts())

fig = plt.figure(figsize= (5,6))
counts = zomato_df['book_table'].value_counts()
ax= fig.gca()
counts.plot.bar(ax=ax, color='green')
plt.title('Book_table'+ ' count distribution')
plt.xlabel('book table')
plt.ylabel('counts')
plt.show()

'''


'''

'''
Observation :

1) As we can see that Most of the restaurants (i.e. 44626 ) has no facility like booking table.
2) and in 6416 restaurants has book table facility.

'''
# Checking how much Ratings are given by peoples to restaurants online delivery
print(zomato_df.head())

fig = plt.figure(figsize=(10,6))
ax= fig.gca()

sns.boxplot(x= zomato_df['online_order'], y= zomato_df['rate'])
plt.show()

'''


'''


'''
Observations = These observation are due to Outliers present in features

1) whether restaurants having online or not the average rating to restaurants given by the customers is 3.70142
2) maximum rating for online delivery of restaurants by people is around 4.7
3) minimum rating for online delivery given by customer is around 2.7
4) And when people go directly to restaurants they give maximum of ratings around 4.4
5) And when people go directly to restaurants they give manimum of ratings around 3.3

'''

# Ploting Violin plot to overcome from outliers
sns.violinplot(x= zomato_df['online_order'], y= zomato_df['rate'])
plt.show()

'''


'''


'''
Actual Observations from rating for restaurants online delivery :

1) whether restaurants having online or not the average rating to restaurants given by the customers is 3.70142
2) maximum rating for online delivery of restaurants by people is around 4.9
3) minimum rating for online delivery given by customer is around 2.1
4) And when people go directly to restaurants they give maximum of ratings around 4.9
5) And when people go directly to restaurants they give minimum of ratings around 1.8

'''
# Checking From which Location of restaurants recieving most online orders
locationwise_online_order= zomato_df.groupby(['location','online_order'])['name'].count()
print(locationwise_online_order)

locationwise_online_order.plot(kind= 'bar',color='cyan', figsize= (16,8))
plt.show()

'''


'''


'''
observations :

1) As we have allready seen that most of the restaurant are present in BTM area and it is obvious things that most of there restaurant will have online delivery facility.
2) In HSR location around 2000 restaurants have online delivery facility.

'''

# Visualizing restaurants types and there ratings
plt.figure(figsize=(16,6))
sns.boxplot(x= zomato_df['listed_in(type)'], y= zomato_df['rate'])
plt.show()

'''


'''

'''
Observations :

1) So Drinks and nightlife Restaurants types has maximum avg rating around 4.2 among other 6 types of restaurants
2) Then Buffet restaurants have avg rating around 4.0
3) worst Rating for restaurants types are Delivery, Desserts, and Dine-out.

'''

# visualizing location wise Restaurant types!
# creating a separate Dataframe of "location" as Row index, and Types of restaurants as (listed _in(type)) with there names as columns

loc_type_df = zomato_df.groupby(['listed_in(type)','location'])['name'].count()

# Creating CSV File
loc_type_df.to_csv('loc_type.csv')
loc_type_df = pd.read_csv('loc_type.csv')
loc_type_df= pd.pivot_table(loc_type_df, values=None, index='location', columns='listed_in(type)',fill_value=0, aggfunc= np.sum)
print(loc_type_df)

# ploting bar plot to get better visualization
loc_type_df.plot(kind = 'bar', color='red', figsize=(30,10))
plt.ylabel('no of restaurants')
plt.show()

'''


'''

# Checking which location got most of votes
# making "location" column and "votes" column in one dataframe
loc_votes_df = zomato_df[['votes','location']]

# there are some duplicates values of number of  restaurants in location
# So Dropping duplicate values
print(loc_votes_df.drop_duplicates())

# summing all the votes for that perticular location
loc_votes_df2=loc_votes_df.groupby(['location'])['votes'].sum()        
loc_votes_df2=loc_votes_df2.to_frame()

#  now sorting the  Data in descending orders of number of votes
loc_votes_df2=loc_votes_df2.sort_values('votes',ascending=False)
print(loc_votes_df2.head())

# Now ploting Bar plot for better visualization
plt.figure(figsize= (27,8))
sns.barplot(x= loc_votes_df2.index,y= loc_votes_df2['votes'] )

# Rotation used to not overlap values
plt.xticks(rotation=90)       
plt.show()

'''



'''


'''

Observations :

1) Here in graph Votes are in 10^6
2) So highest number of votes are from location "Koramangala 5th Block" is 2214083
3) and least number of votes are from "Banaswadi" location

'''

print(zomato_df.head())

# Checking which cuisines got how much votes
# creating dataframe
cuisines_votes_df = zomato_df[['votes','cuisines']]
print(cuisines_votes_df)

# Droping Duplicates
cuisines_votes_df.drop_duplicates()

# Combining cuisines and votes
cuisines_votes_df2 = cuisines_votes_df.groupby(['cuisines'])['votes'].sum()
print(cuisines_votes_df2)

# Converting it into dataframe
cuisines_votes_df2 = cuisines_votes_df2.to_frame()                              ## Converting it into dataframe
print(cuisines_votes_df2.head())

# Sorting the data
cuisines_votes_df2= cuisines_votes_df2.sort_values('votes',ascending=False)
print(cuisines_votes_df2.head())

# Removing first row i.e. "less_num_cuisines" becoz it will get biased visualization
cuisines_votes_df2= cuisines_votes_df2.iloc[1: , :]
print(cuisines_votes_df2.head())

# Plortting Data
plt.figure(figsize=(30,8))
sns.barplot(x=cuisines_votes_df2.index,y=cuisines_votes_df2['votes'])
plt.xticks(rotation=90)
plt.show()

'''


'''

'''
Observations :

1) from above bar plot we see that for "North indian" foods restaurants gets more number of votes i.e. 516310
2) for "north indian chinese" restaurants gets 258225 votes
3) For "mithai" food items restaurants gets lowest votings.

'''