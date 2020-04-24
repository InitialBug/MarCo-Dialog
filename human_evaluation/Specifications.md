# Dialog Human Evaluation Specification

<a name="Tczul"></a>
## 1. completion
the response correctly answers the user query.

- the response correctly answers the **user query**.
- under the above condition, if the response provides **wrong** **information**, the system loses. Also if the response provides extra **useful** **information**, the system wins extra points.


<br />Examples:<br />a.  

| user | any type of college is fine can i just get the postcode , entrance fee , and phone number of [value_count] of them ? |
| --- | --- |
| sys1 | [attraction_name] is located at [attraction_address] , postcode [attraction_postcode] , phone [attraction_phone] . i do not have the entrance fee , but the phone number is [attraction_phone] and postcode is [attraction_postcode] . |
| sys2 | [attraction_name] is free . the phone number is [attraction_phone] and the postcode is [attraction_postcode] . |

=> tie (both correctly answer the user query)<br />
<br />b.  

| user | yes , i am also looking for a multiple sports attraction in the city [attraction_area] . are you aware of any ? |
| --- | --- |
| sys1 | i am sorry , there are no entertainment attractions in the [attraction_area] . would you like to try a different area ? |
| sys2 | there are no multiple sports attractions in the [attraction_area] . would you like to try another area ? |

=> sys2 (sys1 does not answer the user query)<br />
<br />c.  

| user | i am looking for a restaurant in the [restaurant_area] . |
| --- | --- |
| sys1 | there are [value_count] [restaurant_food] restaurant -s in the [restaurant_area] . do you have a specific type of food or price range in mind ? |
| sys2 | there are [value_count] restaurant -s that meet your criteria . what type of food are you looking for ? |

=> tie (sys1 and sys2 both answer the user query and no extra information is provided.)<br />
<br />d.

| user | i need a place to go and should be a museum |
| --- | --- |
| sys1 | there are [value_count] museums in [value_place] . is there a particular area you would like to visit ? |
| sys2 | there are [value_count] museums in the [attraction_area] of town . do you have a specific area in mind ? |

=> tie (the same as c)<br />
<br />e.  

| user | yes i would like to arrive at [train_arriveby] please for [value_count] people , i need the reference number |
| --- | --- |
| sys1 | i have booked you [value_count] seats on the [train_trainid] . the total fee is [train_price] gbp payable at the station . your reference number is [train_reference] . is there anything else i can help you with ? |
| sys2 | booking was successful , the total fee is [train_price] gbp payable at the station . reference number is [train_reference] . is there anything else i can help you with ? |

=> sys1 (sys1 and sys2 both correctly answers the query, and sys1 provides extra information)<br />
<br />f.

| user | that works . i am also looking for an attratiction that is college type . |
| --- | --- |
| sys | there are [value_count] different colleges in [value_place] . did you wish to go to a certain part of town ? |
| user | no particular area , what s the first [value_count] on the list ? |
| sys1 | i would recommend [attraction_name] . would you like their phone number ? |
| sys2 | i would recommend [attraction_name] . it s free to get in . would you like more information ? |

=> sys2 (sys2 provide extra useful information -- the recommendation reason -- it 's free to get in)<br />
<br />


---



<a name="nNVYx"></a>
## 2. readability
the response is fluent, natural and consistent.

- fluent: there are no repetitions in one sentence.
- consistent: there are no repeated or conflicted information between response and dialog history.


<br />
<br />Examples:<br />a.  

| user | any type of college is fine can i just get the postcode , entrance fee , and phone number of [value_count] of them ? |
| --- | --- |
| sys1 | [attraction_name] is located at [attraction_address] , postcode [attraction_postcode] , phone [attraction_phone] . i do not have the entrance fee , but the phone number is [attraction_phone] and postcode is [attraction_postcode] . |
| sys2 | [attraction_name] is free . the phone number is [attraction_phone] and the postcode is [attraction_postcode] . |

=> sys2 (repetition in sys1)<br />
<br />c.  

| user | hello , i am looking to book a train that will be departing [train_departure] after [train_leaveat] . |
| --- | --- |
| sys1 | there is a train leaving [train_departure] on [train_day] at [train_leaveat] . where would you like to depart ? |
| sys2 | there are [value_count] trains leaving [train_departure] on [train_day] . what is your destination and what day would you like to travel ? |

=> sys1 (inconsistent, destination is already known in user's query.)<br />
<br />d.  

| user | book it for [value_count] people , [value_count] nights , starting [value_day] . give me the reference number . |
| --- | --- |
| sys1 | booking was successful . reference number is [hotel_reference] . is there anything else i can help you with ? |
| sys2 | i was able to book you at the [hotel_name] for [value_count] nights . your reference number is [hotel_reference] . is there anything else i can help you with ? |

=> tie (although sys2 provides more information, sys1 and sys2 both read fluent and consistant.)<br />
<br />e.  

| user | thanks for your help with the hotel . how about a restaurant in the [restaurant_area] that serves [restaurant_food] food ? |
| --- | --- |
| sys1 | there are [value_count] [restaurant_food] restaurant -s in the [restaurant_area] . [restaurant_name] , [restaurant_name] , and [restaurant_name] . would you like to book a table there ? |
| sys2 | there are [value_count] [restaurant_pricerange] [restaurant_food] restaurant -s in the [restaurant_area] . i recommend the [restaurant_name] . would you like me to book a table for you ? |

=> tie<br />
<br />f.  

| user | okay do they have free wifi ? |
| --- | --- |
| sys1 | yes , they do have free wifi . would you like me to book it for you ? |
| sys2 | yes , it does have free wifi and parking . would you like me to book it for you ? |

=> tie (it both OK to use `they,it or [hotel_name]).`

f. 

| user | does the museum have an entrance fee ? |
| --- | --- |
| sys1 | yes , it is free . is there anything else i can help you with ? |
| sys2 | [attraction_name] is free . would you like more information ? |

=> sys2 (sys1: yes, it is free . -> no, it is free.)<br />
<br />PS. `thank you for using the [value_place] towninfo [restaurant_area] .` is a natural sentence.<br />=> `thank you for using the cambridge towninfo center.`
