multinews_dialogue1 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topics: 'unemployment rate, jobs report, economy'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user's documents and find relevant people who may have more information")
>>> EnterpriseSearch.search_documents(query="unemployment rate, jobs report, economy")
[]
>>> EnterpriseSearch.search_relevant_people(query="unemployment rate, jobs report, economy")
{
   "hannah": "Knows about the following topics:\nhiring, unemployment rate, job growth, economy, consumer spending\nGaithersburg, Maryland, jet crash, Montgomery County, FAA",
   "parker": "Knows about the following topics:\nShelly Sterling, LA Clippers sale, divorce\nGaithersburg, plane crash, Marie Gemmell, Michael Rosenberg, Montgomery County Air Park",
   "eden": "Knows about the following topics:\njobs, unemployment rate, economy"
}
>>> Reflection.thought("The current user has no documents. Hannah and Eden have information about the economy, so I should reach out to them.")
>>> Enterprise.send_message("alice", "Hi Alice, I couldn't find enough information in your documents to compile that information directly. I can reach out to Hannah and Eden who may have more information. Sound good?")
>>> System.finish()

# Event: Alice Anastasiou says: sure!
>>> Reflection.thought("I will reach out to Hannah to ask for information about the economy")
>>> Enterprise.resolve_person("hannah")
{"user_id": "hannah", "email": "hannah@company.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, do you have any information about the unemployment rate, jobs report, or economy?", title="Information about economy")
>>> System.finish()

# Event: Hannah Jarvis says: Sure, here's what I have: "employers pulled back sharply on hiring last month , a reminder that the u.s. economy may not be growing fast enough to sustain robust job growth . the unemployment rate dipped , but mostly because more americans stopped
looking for work .     the labor department says the economy added 120,000 jobs in march , down from more than 200,000 in each of the previous three months .     the unemployment rate fell to 8.2 percent , th
e lowest since january 2009. the rate dropped because fewer people searched for jobs . the official unemployment tally only includes those seeking work .     the economy has added 858,000 jobs since december
_ the best four months of hiring in two years . but federal reserve chairman ben bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending ."
>>> Reflection.thought("I will reach out to Eden to ask for information about the economy")
>>> Enterprise.resolve_person("Eden")
{"user_id": "eden", "email": "eden@company.com", "full_name": "Eden Berhe"}
>>> Enterprise.send_message("cassie", "Hi Eden, do you have any information about the unemployment rate, jobs report, or economy?", title="Information about economy")
>>> System.finish()

# Event: Eden Berhe says: Yes, I have an article with the following contents: "national archives     yes , it \u2019 s that time again , folks . it \u2019 s the first friday of the month , when for one ever-so-brief moment the interests of wall street , washington and main street are all aligned on one thing : jobs .     a fresh update on the u.s. employment situation for january hits the wires at 8 : 30 a.m. new york time offering one of the most important snapshots on how the economy fared during the previous month . expectations are for 203,000 new jobs to be created , according to economists polled by dow jones newswires , compared to 227,000 jobs added in february . the unemployment rate is expected to hold steady at 8.3 % .     here at marketbeat hq , we \u2019 ll be offering color commentary before and after the data crosses the wires . feel free to weigh-in yourself , via the comments section . and while you \u2019 re here , why don \u2019 t you sign up to  .     enjoy the show"
>>> Reflection.thought("I have enough information to compile the summary. I will send a message to Alice, with the summary enclosed in [[[ brackets ]]].")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[ the unemployment rate dropped to 8.2 % last month , but the economy only added 120,000 jobs , when 203,000 new jobs had been predicted , according to today ' s jobs report . reaction on the wall street journal ' s marketbeat blog was swift : \" woah ! ! ! bad number . \" the unemployment rate , however , is better news ; it had been expected to hold steady at 8.3 % . but the ap notes that the dip is mostly due to more americans giving up on seeking employment ]]]", title="Economy summary")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That would be all.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


multinews_dialogue2 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topic: 'fracking fluid, arlington, texas, vantage energy, gas leak'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user's documents and find relevant people who may have more information")
>>> EnterpriseSearch.search_documents(query="fracking fluid, arlington, texas, vantage energy, gas leak")
[]
>>> EnterpriseSearch.search_relevant_people(query="moon")
{
  "bhushan": "Knows about the following topics:\nDiane Rehm, Bernie Sanders, dual citizenship, apology, NPR\ncorporal punishment, John B. King Jr., education, school discipline, Obama administration\nJohn King, corporal punishment, school discipline, students of color, students with disabilities\nArlington, fracking, Vantage Energy, spill, toxic substances",
  "farshid": "Knows about the following topics:\nBrentwood, wheelchair ramp, homeowners association, Michael Broadnax, Tennessee Human Rights Commission\nwarczone, warcs, internet archive, wayback machine, websites\ncorporal punishment, school discipline, National Women's Law Center, ACLU, American Academy of Pediatrics\ngas leak, Arlington, Vantage Energy, fracking, senior center",
  "parker": "Knows about the following topics:\nArlington, gas well, fracking fluid, Vantage Energy, leak",
  "hannah": "Knows about the following topics:\nTim Hunt, Nobel Prize, sexism, science, women\nMark Kirk, Lindsey Graham, South Side, Chicago, Mary Mitchell"
}
>>> Reflection.thought("The current user has no documents. Bhushan, Farshid and Parker have information about the Vantage Energy leak, so I should reach out to them.")
>>> Enterprise.send_message("alice", "Hi Alice, it looks like Bhushan, Farshid and Parker have relevant information, so I can reach out to them to produce a summary. Is that OK?")
>>> System.finish()

# Event: Alice Anastasiou says: Yes, that sounds great!
>>> Reflection.thought("I will reach out to Bhushan to ask him for information about the Vantage Energy leak")
>>> Enterprise.resolve_person("bhushan")
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage Energy leak")
>>> System.finish()

# Event: Bhushan Magar says: Sure, here's what I have: "vantage energy operates the natural gas drilling site on the grounds of lake arlington baptist church . ( photo : wfaa )     arlington \u2014 two months ago , 100 homes in arlington had to be evacuated as fracking fluid spilled out of a drilling site onto the city streets .     now we know officially what happened , why it happened , and why arlington officials are blaming the drilling company for \" unacceptable behavior . \"     a series of video recordings obtai ned by news 8 shows the scene behind the walls of a fracking site 600 feet from a cluster of homes in the state ' s seventh largest city . in the incident , 42,800 gallons of fracking fluid \u2014 boiling up from thousands of feet underground \u2014 spewed into the streets and into arlington storm sewers and streams .     wfaa obtained this video that shows a leak of fracking fluid from an arlington drilling site ( photo : wfaa )     four attempts and 24 hours later , experts were finally able to plug the natural gas well .     nearby residents and arlington officials feared the worst . now , two months later , fire officials have concluded their investigation .     \" clearly there was a release of unpermitted materials into the stormwater system , \" said arlington fire chief don crowson as he addressed arlington city council members on tuesday .     the good news , according to crowson : despite numerous toxic substances being released into the environment , tests show it was not in amounts that did significant damage to the environment .     the bad news ? he said the drilling company mishandled the spill .     \" for my concerns , the main issue i articulated to you a few months ago was the delayed notification of 911 , \" crowson said . \" it ' s not acceptable . \"     according to the report , vantage energy first contacted 911 nearly two hours after fracking water first started to spill . what ' s more , the call to 911 cam e not from the site , but from corporate headquarters in pennsylvania .     \" this is unacceptable behavior , \" said city council member robert rivera . \" the citizens of arlington do not appreciate the la ck of ability to control the site . \"     labc gas well leak ( photo : wfaa )     the official causeof the spill at a site adjacent to lake arlington baptist church is listed as equipment failure . vantage e nergy was issued a citation and has agreed to reimburse the city $ 84,000 .     but this was not included in the city ' s report : records uncovered by news 8 of another 1,500-gallon spill at the same site on e month earlier .     despite numerous toxic substances being released into the environment , tests show it was not in amounts that did significant damage to the environment .     arlington resident kim feil said the two incidents one month apart reinforce her fears that drilling so close to homes is not safe .     \" i just assumed this was a residential area and it would be free from industrial hazardous operat ions , \" feil said . \" now we see it ' s not . \"     in the meantime , drilling operations remain shut down and will not resume until the city does a final inspection and the folks across the street and th ose affected are given official notification .      http://on.wfaa.com/1mjmrs"
>>> Reflection.thought("I will reach out to Farshid to ask him for information about the Vantage Energy leak")
>>> Enterprise.resolve_person("farshid")
{"user_id": "farshid", "email": "farshid@company.com", "full_name": "Farshid Kamangar"}
>>> Enterprise.send_message("farshid", "Hi Farshid, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage energy leak")
>>> System.finish()

# Event: Farshid Kamangar says: Yes, I have an article with the following contents: "the fire department has taken a gas well operator to task and imposed rules to make sure a leak like the one that occurred in southwest arlington on april 11 do esn \u2019 t happen again , top fire officials told the city council at an afternoon work session tuesday .     a pipe sprung a softball-size hole at the vantage energy well site at 3016 little road , allowin g nearly 43,000 gallons of fracking water and chemicals to gush into the city storm-water system and sparking fears that natural gas would follow , assistant fire chief jim self said .     more than 100 famil ies living within 1/8 mile were evacuated as a precaution .     \u201csome were displaced as long as 21 hours , \u201d self said .     never miss a local story . sign up today for a free 30 day free trial of unlimited digital access . subscribe now     no gas leaked , but vantage employees , trying to fix the problem themselves , waited two hours to call 911.     \u201cthat \u2019 s unacceptable behavior , \u201d councilman robert rivera said .     self and fire chief don crowson assured the council that they have taken steps to clarify the city \u2019 s expectations in such emergencies . step 1 : call 911 before doi ng anything else .     \u201cwe \u2019 re not kidding around about the 911 issue , \u201d crowson said during a break . \u201cit was a very serious situation . it could have ended in a bad outcome . two hours \u2019 advance notice could have helped alot . luckily , we were prepared and we worked well with [ vantage ] , and we were able to resolve the issue. \u201d     the pipe began leaking about 1 p.m. it was ca rrying water and solvents that had been used , under high pressure , to fracture shale and release its natural gas . at any moment , the frack-water leak could have become a gas leak .     stopping the leak t ook almost 24 hours .     city officials praised vantage officials for their cooperation . city manager trey yelverton said vantage hasn \u2019 t been the only focus . the city \u2019 s expectations have been communicated to all drilling operators in arlington .     in other business , the council told parks officials to work both a senior center and a \u201cmultigenerational \u201d activities center into the ong oing update of the parks master plan .     parks director lemuel randolph estimated the senior center \u2019 s cost at $ 25 million and the all-ages facility at $ 40 million . he said the master plan was virt ually finished in november when a new senior center became a priority .     about 25 older residents in red t-shirts attended the afternoon session to lend silent support for a stand-alone senior center . the n at the evening council meeting , they weren \u2019 t so silent . several walked to the lectern at the end of the meeting to lobby .     elva roy , head of age-friendly arlington action brigade , asked that the city dedicate the last quarter-cent available in the city sales tax to raising money for a senior center , and put the issue on the november election ballot .     seniors now use portions of two aging act ivities center , called eunice and new york . the seniors want something like the summit , the $ 23 million 50-and-over senior center in grand prairie that opened in 2010 , roy said .     \u201cwhen you walk into the summit , it \u2019 s just so tranquil , \u201d she said ."
>>> Reflection.thought("I will reach out to Parker to ask him for information about the Vantage Energy leak")
>>> Enterprise.resolve_person("parker")
{"user_id": "parker", "email": "parker@company.com", "full_name": "Parker McLean"}
>>> Enterprise.send_message("parker", "Hi Parker, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage Energy leak")
>>> System.finish()

# Event: Parker McLean says: Yep, here's my article: "arlington fire officials indicated the site of a gas well that leaked thousands of gallons of fracking fluid back in april could be close to reopening . ( published tuesday , june 16 , 2015 )     an arlington gas well site that leaked thousands of gallons of fracking fluid in april could soon resume drilling .     all operations at vantage energy ' s lake arlington baptist church site along little road have been suspended since that leak occurred .     the company said a well head component malfunctioned , causing a back flow of fracking fluid to spill out . no natural gas leaked from the well , but the incident forced dozens of families in the area to evacuate their homes .     on tuesday , arlington fire chief don crowson gave the city council an update on the cleanup efforts .     92-year-old woman arrested in denton fracking protest     a 92-year-old mother and son were the latest arrests in denton \u2019 s fracking protests tuesday ; only that mother also happens to be a great grandmother . ( published tuesday , june 16 , 2015 )     ( tue jun 16 16 : 40 : 54 pdt 2015 $ __output )     \" i will give vantage credit for this , \" said crowson . \" they ' ve owned it . they ' ve owned the responsibility for it . \"     through the course of their investigation , fire officials determined more than 42,000 gallons of fracking fluid escaped into the city ' s storm water system . they don ' t believe the city ' s water supply or the public are in any danger .     \" arlington environmental , the texas commission on environmental quality and the epa tested the water and soil samples in several locations and found pollutant material below levels that will require further remediation , \" said assistant fire chief jim self .     crowson said vantage has identified the faulty component that caused the well head to malfunction and has taken steps to fix the problem on that well and others .     the company has also paid the city more than $ 82,000 to reimburse taxpayers for the response to the leak . that doesn ' t include additional fines vantage had to pay for three citations the city issued following the incident .     \" our relationship with vantage has been very good historically , \" said crowson .     crowson did not mince words , though , when he discussed vantage ' s decision to wait nearly two hours before reporting the leak to 911. he called the company ' s actions \" unacceptable \" and said he ' s met with vantage and other energy companies to make sure they understand that .     \" call the fire department immediately , \" said crowson . \" that ' s been carefully and directly communicated . \"     in an email to nbc 5 , a spokesperson for vantage said , \" we ' ve worked collaboratively with the fire department on revised notification procedures . \"     crowson said the site has to pass one last inspection before drilling can resume . when that happens , he said the public will be notified . "
>>> Reflection.thought("I have enough information to compile the summary. I will send a message to Alice, with the summary enclosed in [[[ brackets ]]].")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[a massive leak of fracking fluid poured into the streets of arlington , texas , two months ago and forced the evacuation of a hundred homes . now city officials have taken vantage energy to task for its \" unacceptable \" handling of the 43,000-gallon spill , wfaa reports . during a city council meeting yesterday , it emerged that vantage had taken nearly two hours to call 911 despite the risk of a gas leak . \" this is unacceptable behavior , \" says an arlington city council member . according to fire chief don crowson , the two-hour delay was no joke : \" we \u2019 re not kidding around about the 911 issue , \" he tells the star-telegram during a break in city council . \" it could have ended in a bad outcome . two hours \u2019 advance notice could have helped a lot . \" still , officials say the environmental damage was not extensive and vantage has been cooperative . so what happened , exactly ? according to the city ' s report , a vantage well site sprung a leak on april 11 , which allowed fracking water and chemicals to boil up into arlington ' s streets , storm sewers , and streams . because the fracking fluid had been fracturing shale and freeing gas under high pressure , natural gas could have leaked at any time . now wfaa says a 1,500-gallon spill occurred at the same location a month before , and nbc-dallas/fort worth reports that the site is close to reopening . \" i just assumed this was a residential area and it would be free from industrial hazardous operations , \" says a resident after hearing about the earlier spill . \" now we see it ' s not . \"]]]", title="Vantage Energy leak summary")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That's all I need.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""

multinews_dialogue3 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topics: 'Howard Stern, ice bucket challenge, Casey Kasem'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user's documents and find relevant people who may have more information")
>>> EnterpriseSearch.search_documents(query="Howard Stern, ice bucket challenge, Casey Kasem")
[]
>>> EnterpriseSearch.search_relevant_people(query="Howard Stern, ice bucket challenge, Casey Kasem")
{
  "cassie": "Knows about the following topics:\ndine-and-dash, dating app, bumble, scam, blind date\nBritish Airways, Mr Justice Peter Smith, lost luggage, Emerald Supplies Ltd v British Airways",
  "bhushan": "Knows about the following topics:\ndining and dashing, Paul Gonzales, Bumble, BJ's Brewhouse, serial check-skipper\nHoward Stern, ALS ice bucket challenge, Matt Lauer, Jennifer Aniston",
  "hannah": "Knows about the following topics:\nHoward Stern, ALS, Ice Bucket Challenge\njudge, bias, British Airways, luggage, court case",
  "farshid": "Knows about the following topics:\nBritish Airways, lost luggage, Mr Justice Peter Smith, judicial conduct authorities, bias",
  "parker": "Knows about the following topics:\n",
  "dewei": "Knows about the following topics:\nBritish Airways, lost luggage, Mr Justice Peter Smith, bias, court case"
}
>>> Reflection.thought("Bhushan and Hannah have articles about Howard Stern, so I should reach out to them.")
>>> Enterprise.send_message("alice", "Hi Alice, I can reach out to Bhushan and Hannah to produce a summary. Is that OK?")
>>> System.finish()

# Event: Alice Anastasiou says: Yes, that sounds great!
>>> Reflection.thought("I will reach out to Bhushan to ask for information about the ice bucket challenge")
>>> Enterprise.resolve_person("Bhushan")
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, do you have any information Howard Stern, the ice bucket challenge, or Casey Kasem?", title="Information about ice bucket challenge")
>>> System.finish()

# Event: Bhushan Magar says: Sure, here's an article I found: "wtf ? !     howard stern recently completed the als ice bucket challenge and shared a video of the do-gooder act on youtube . while stern doing the bone-chilling charitable act is nothing out of the ordinary , you may be scratching your head when you hear who he nominates to undertake the challenge next .     \" hey everybody , it ' s howard stern ready to take the ice bucket challenge , \" a shirtless stern says in the video . \" i ' m accepting the challenge of ... who challenged me ? matt lauer and jennifer aniston . "
>>> Reflection.thought("I will reach out to Hannah to ask for information about the ice bucket challenge")
>>> Enterprise.resolve_person("Hannah")
{"user_id": "hannah", "email": "hannah@company.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, do you have any information about Howard Stern, the ice bucket challenge, or Casey Kasem?", title="Information about ice bucket challenge")
>>> System.finish()

# Event: Hannah Jarvis says: Yes, I have the following article: "after both jennifer aniston and matt lauer nominated him , howard stern finally accepted the ice bucket challenge - and you won ' t believe who he nominated !     remember , all this ice bucket nonsense is for a good cause - donate to the als association and help strike out als !     and watch beth stern take the ice bucket challenge too"
>>> Reflection.thought("I have enough information to compile the summary. I will send a message to Alice, with the summary enclosed in [[[ brackets ]]].")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[howard stern has nominated an interesting trio to complete the ice bucket challenge after him . trouble is , one of them is dead , reports e ! online . in a bizarre video posted to youtube yesterday , a bare-chested stern plops an ice cube into a shot glass of water , then pours it over his head , immediately succumbing to mock shivers before hopping off camera . ]]]", title="Ice bucket challenge summary")
>>> System.finish()

# Event: Alice Anastasiou says: Perfect---that's all for now.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""

multinews_none_dialogue1 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topics: 'Howard Stern, ice bucket challenge, Casey Kasem'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will now produce a summary on the topics 'Howard Stern, ice bucket challenge, Casey Kasem'.")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[howard stern has nominated an interesting trio to complete the ice bucket challenge after him . trouble is , one of them is dead , reports e ! online . in a bizarre video posted to youtube yesterday , a bare-chested stern plops an ice cube into a shot glass of water , then pours it over his head , immediately succumbing to mock shivers before hopping off camera . ]]]", title="Ice bucket challenge summary")
>>> System.finish()

# Event: Alice Anastasiou says: Perfect---that's all for now.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""

multinews_none_dialogue2 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topic: 'fracking fluid, arlington, texas, vantage energy, gas leak'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will now produce a summary on the topics 'fracking fluid, arlington, texas, vantage energy, gas leak'.")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[a massive leak of fracking fluid poured into the streets of arlington , texas , two months ago and forced the evacuation of a hundred homes . now city officials have taken vantage energy to task for its \" unacceptable \" handling of the 43,000-gallon spill , wfaa reports . during a city council meeting yesterday , it emerged that vantage had taken nearly two hours to call 911 despite the risk of a gas leak . \" this is unacceptable behavior , \" says an arlington city council member . according to fire chief don crowson , the two-hour delay was no joke : \" we \u2019 re not kidding around about the 911 issue , \" he tells the star-telegram during a break in city council . \" it could have ended in a bad outcome . two hours \u2019 advance notice could have helped a lot . \" still , officials say the environmental damage was not extensive and vantage has been cooperative . so what happened , exactly ? according to the city ' s report , a vantage well site sprung a leak on april 11 , which allowed fracking water and chemicals to boil up into arlington ' s streets , storm sewers , and streams . because the fracking fluid had been fracturing shale and freeing gas under high pressure , natural gas could have leaked at any time . now wfaa says a 1,500-gallon spill occurred at the same location a month before , and nbc-dallas/fort worth reports that the site is close to reopening . \" i just assumed this was a residential area and it would be free from industrial hazardous operations , \" says a resident after hearing about the earlier spill . \" now we see it ' s not . \"]]]", title="Vantage Energy leak summary")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That's all I need.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


multinews_all_dialogue1 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topics: 'unemployment rate, jobs report, economy'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user's documents and find relevant people who may have more information")
>>> EnterpriseSearch.search_documents(query="unemployment rate, jobs report, economy")
[]
>>> EnterpriseSearch.search_relevant_people(query="unemployment rate, jobs report, economy")
{
   "hannah": "Knows about the following topics:\nhiring, unemployment rate, job growth, economy, consumer spending\nGaithersburg, Maryland, jet crash, Montgomery County, FAA",
   "parker": "Knows about the following topics:\nShelly Sterling, LA Clippers sale, divorce\nGaithersburg, plane crash, Marie Gemmell, Michael Rosenberg, Montgomery County Air Park",
   "eden": "Knows about the following topics:\njobs, unemployment rate, economy"
}
>>> Reflection.thought("I will message all users returned by the search procedure. This is Hannah, Parker, and Eden.")
>>> Enterprise.send_message("alice", "Hi Alice, I couldn't find enough information in your documents to compile that information directly. I can reach out to Hannah and Eden who may have more information. Sound good?")
>>> System.finish()

# Event: Alice Anastasiou says: sure!
>>> Reflection.thought("I will reach out to Hannah, Parker, and Eden to ask for information about the economy")
>>> Enterprise.resolve_person("hannah")
{"user_id": "hannah", "email": "hannah@company.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, do you have any information about the unemployment rate, jobs report, or economy?", title="Information about economy")
>>> System.finish()

# Event: Hannah Jarvis says: Sure, here's what I have: "employers pulled back sharply on hiring last month , a reminder that the u.s. economy may not be growing fast enough to sustain robust job growth . the unemployment rate dipped , but mostly because more americans stopped looking for work .     the labor department says the economy added 120,000 jobs in march , down from more than 200,000 in each of the previous three months .     the unemployment rate fell to 8.2 percent , th e lowest since january 2009. the rate dropped because fewer people searched for jobs . the official unemployment tally only includes those seeking work .     the economy has added 858,000 jobs since december _ the best four months of hiring in two years . but federal reserve chairman ben bernanke has cautioned that the current hiring pace is unlikely to continue without more consumer spending ."
>>> Reflection.thought("I still need to contact Parker and Eden. I will contact Parker first.")
>>> Enterprise.resolve_person("Parker")
{"user_id": "parker", "email": "parker@company.com", "full_name": "Parker McLean"}
>>> Enterprise.send_message("parker", "Hi Parker, do you have any information about the unemployment rate, jobs report, or economy?", title="Information about economy")
>>> System.finish()

# Event: Parker McLean says: Sorry, I'm afraid I don't have any information on that topic.
>>> Reflection.thought("I still need to contact Eden. I will contact her now.")
>>> Enterprise.resolve_person("Eden")
{"user_id": "eden", "email": "eden@company.com", "full_name": "Eden Berhe"}
>>> Enterprise.send_message("eden", "Hi Eden, do you have any information about the unemployment rate, jobs report, or economy?", title="Information about economy")
>>> System.finish()

# Event: Eden Berhe says: Yes, I have an article with the following contents: "national archives     yes , it \u2019 s that time again , folks . it \u2019 s the first friday of the month , when for one ever-so-brief moment the interests of wall street , washington and main street are all aligned on one thing : jobs .     a fresh update on the u.s. employment situation for january hits the wires at 8 : 30 a.m. new york time offering one of the most important snapshots on how the economy fared during the previous month . expectations are for 203,000 new jobs to be created , according to economists polled by dow jones newswires , compared to 227,000 jobs added in february . the unemployment rate is expected to hold steady at 8.3 % .     here at marketbeat hq , we \u2019 ll be offering color commentary before and after the data crosses the wires . feel free to weigh-in yourself , via the comments section . and while you \u2019 re here , why don \u2019 t you sign up to  .     enjoy the show"
>>> Reflection.thought("I have enough information to compile the summary. I will send a message to Alice, with the summary enclosed in [[[ brackets ]]].")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[ the unemployment rate dropped to 8.2 % last month , but the economy only added 120,000 jobs , when 203,000 new jobs had been predicted , according to today ' s jobs report . reaction on the wall street journal ' s marketbeat blog was swift : \" woah ! ! ! bad number . \" the unemployment rate , however , is better news ; it had been expected to hold steady at 8.3 % . but the ap notes that the dip is mostly due to more americans giving up on seeking employment ]]]", title="Economy summary")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That would be all.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


multinews_all_dialogue2 = """### Example ###

# Event: Alice Anastasiou says: I'm interested in compiling a summary on the topic: 'fracking fluid, arlington, texas, vantage energy, gas leak'.
>>> Reflection.thought("I will first resolve the primary user to get the right contact information")
>>> Enterprise.resolve_primary_user()
{"user_id": "alice", "email": "alice@company.com", "full_name": "Alice Anastasiou"}
>>> Reflection.thought("I will search user's documents and find relevant people who may have more information")
>>> EnterpriseSearch.search_documents(query="fracking fluid, arlington, texas, vantage energy, gas leak")
[]
>>> EnterpriseSearch.search_relevant_people(query="moon")
{
  "bhushan": "Knows about the following topics:\nDiane Rehm, Bernie Sanders, dual citizenship, apology, NPR\ncorporal punishment, John B. King Jr., education, school discipline, Obama administration\nJohn King, corporal punishment, school discipline, students of color, students with disabilities\nArlington, fracking, Vantage Energy, spill, toxic substances",
  "farshid": "Knows about the following topics:\nBrentwood, wheelchair ramp, homeowners association, Michael Broadnax, Tennessee Human Rights Commission\nwarczone, warcs, internet archive, wayback machine, websites\ncorporal punishment, school discipline, National Women's Law Center, ACLU, American Academy of Pediatrics\ngas leak, Arlington, Vantage Energy, fracking, senior center",
  "parker": "Knows about the following topics:\nArlington, gas well, fracking fluid, Vantage Energy, leak",
  "hannah": "Knows about the following topics:\nTim Hunt, Nobel Prize, sexism, science, women\nMark Kirk, Lindsey Graham, South Side, Chicago, Mary Mitchell"
}
>>> Reflection.thought("I will message all users returned by the search procedure. This is Bhushan, Farshid, Parker, and Hannah.")
>>> Enterprise.send_message("alice", "Hi Alice, it looks like Bhushan, Farshid, Parker and Hannah have information, so I can reach out to them to produce a summary. Is that OK?")
>>> System.finish()

# Event: Alice Anastasiou says: Yes, that sounds great!
>>> Reflection.thought("I will reach out to Bhushan to ask him for information about the Vantage Energy leak")
>>> Enterprise.resolve_person("bhushan")
{"user_id": "bhushan", "email": "bhushan@company.com", "full_name": "Bhushan Magar"}
>>> Enterprise.send_message("bhushan", "Hi Bhushan, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage Energy leak")
>>> System.finish()

# Event: Bhushan Magar says: Sure, here's what I have: "vantage energy operates the natural gas drilling site on the grounds of lake arlington baptist church . ( photo : wfaa )     arlington \u2014 two months ago , 100 homes in arlington had to be evacuated as fracking fluid spilled out of a drilling site onto the city streets .     now we know officially what happened , why it happened , and why arlington officials are blaming the drilling company for \" unacceptable behavior . \"     a series of video recordings obtai ned by news 8 shows the scene behind the walls of a fracking site 600 feet from a cluster of homes in the state ' s seventh largest city . in the incident , 42,800 gallons of fracking fluid \u2014 boiling up from thousands of feet underground \u2014 spewed into the streets and into arlington storm sewers and streams .     wfaa obtained this video that shows a leak of fracking fluid from an arlington drilling site ( photo : wfaa )     four attempts and 24 hours later , experts were finally able to plug the natural gas well .     nearby residents and arlington officials feared the worst . now , two months later , fire officials have concluded their investigation .     \" clearly there was a release of unpermitted materials into the stormwater system , \" said arlington fire chief don crowson as he addressed arlington city council members on tuesday .     the good news , according to crowson : despite numerous toxic substances being released into the environment , tests show it was not in amounts that did significant damage to the environment .     the bad news ? he said the drilling company mishandled the spill .     \" for my concerns , the main issue i articulated to you a few months ago was the delayed notification of 911 , \" crowson said . \" it ' s not acceptable . \"     according to the report , vantage energy first contacted 911 nearly two hours after fracking water first started to spill . what ' s more , the call to 911 cam e not from the site , but from corporate headquarters in pennsylvania .     \" this is unacceptable behavior , \" said city council member robert rivera . \" the citizens of arlington do not appreciate the la ck of ability to control the site . \"     labc gas well leak ( photo : wfaa )     the official causeof the spill at a site adjacent to lake arlington baptist church is listed as equipment failure . vantage e nergy was issued a citation and has agreed to reimburse the city $ 84,000 .     but this was not included in the city ' s report : records uncovered by news 8 of another 1,500-gallon spill at the same site on e month earlier .     despite numerous toxic substances being released into the environment , tests show it was not in amounts that did significant damage to the environment .     arlington resident kim feil said the two incidents one month apart reinforce her fears that drilling so close to homes is not safe .     \" i just assumed this was a residential area and it would be free from industrial hazardous operat ions , \" feil said . \" now we see it ' s not . \"     in the meantime , drilling operations remain shut down and will not resume until the city does a final inspection and the folks across the street and th ose affected are given official notification .      http://on.wfaa.com/1mjmrs"
>>> Reflection.thought("I still need to contact Farshid, Parker, and Hannah. I will contact Farshid first.")
>>> Enterprise.resolve_person("farshid")
{"user_id": "farshid", "email": "farshid@company.com", "full_name": "Farshid Kamangar"}
>>> Enterprise.send_message("farshid", "Hi Farshid, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage energy leak")
>>> System.finish()

# Event: Farshid Kamangar says: Yes, I have an article with the following contents: "the fire department has taken a gas well operator to task and imposed rules to make sure a leak like the one that occurred in southwest arlington on april 11 do esn \u2019 t happen again , top fire officials told the city council at an afternoon work session tuesday .     a pipe sprung a softball-size hole at the vantage energy well site at 3016 little road , allowin g nearly 43,000 gallons of fracking water and chemicals to gush into the city storm-water system and sparking fears that natural gas would follow , assistant fire chief jim self said .     more than 100 famil ies living within 1/8 mile were evacuated as a precaution .     \u201csome were displaced as long as 21 hours , \u201d self said .     never miss a local story . sign up today for a free 30 day free trial of unlimited digital access . subscribe now     no gas leaked , but vantage employees , trying to fix the problem themselves , waited two hours to call 911.     \u201cthat \u2019 s unacceptable behavior , \u201d councilman robert rivera said .     self and fire chief don crowson assured the council that they have taken steps to clarify the city \u2019 s expectations in such emergencies . step 1 : call 911 before doi ng anything else .     \u201cwe \u2019 re not kidding around about the 911 issue , \u201d crowson said during a break . \u201cit was a very serious situation . it could have ended in a bad outcome . two hours \u2019 advance notice could have helped alot . luckily , we were prepared and we worked well with [ vantage ] , and we were able to resolve the issue. \u201d     the pipe began leaking about 1 p.m. it was ca rrying water and solvents that had been used , under high pressure , to fracture shale and release its natural gas . at any moment , the frack-water leak could have become a gas leak .     stopping the leak t ook almost 24 hours .     city officials praised vantage officials for their cooperation . city manager trey yelverton said vantage hasn \u2019 t been the only focus . the city \u2019 s expectations have been communicated to all drilling operators in arlington .     in other business , the council told parks officials to work both a senior center and a \u201cmultigenerational \u201d activities center into the ong oing update of the parks master plan .     parks director lemuel randolph estimated the senior center \u2019 s cost at $ 25 million and the all-ages facility at $ 40 million . he said the master plan was virt ually finished in november when a new senior center became a priority .     about 25 older residents in red t-shirts attended the afternoon session to lend silent support for a stand-alone senior center . the n at the evening council meeting , they weren \u2019 t so silent . several walked to the lectern at the end of the meeting to lobby .     elva roy , head of age-friendly arlington action brigade , asked that the city dedicate the last quarter-cent available in the city sales tax to raising money for a senior center , and put the issue on the november election ballot .     seniors now use portions of two aging act ivities center , called eunice and new york . the seniors want something like the summit , the $ 23 million 50-and-over senior center in grand prairie that opened in 2010 , roy said .     \u201cwhen you walk into the summit , it \u2019 s just so tranquil , \u201d she said ."
>>> Reflection.thought("I still need to contact Parker and Hannah. I will contact Parker first.")
>>> Enterprise.resolve_person("parker")
{"user_id": "parker", "email": "parker@company.com", "full_name": "Parker McLean"}
>>> Enterprise.send_message("parker", "Hi Parker, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage Energy leak")
>>> System.finish()

# Event: Parker McLean says: Yep, here's my article: "arlington fire officials indicated the site of a gas well that leaked thousands of gallons of fracking fluid back in april could be close to reopening . ( published tuesday , june 16 , 2015 )     an arlington gas well site that leaked thousands of gallons of fracking fluid in april could soon resume drilling .     all operations at vantage energy ' s lake arlington baptist church site along little road have been suspended since that leak occurred .     the company said a well head component malfunctioned , causing a back flow of fracking fluid to spill out . no natural gas leaked from the well , but the incident forced dozens of families in the area to evacuate their homes .     on tuesday , arlington fire chief don crowson gave the city council an update on the cleanup efforts .     92-year-old woman arrested in denton fracking protest     a 92-year-old mother and son were the latest arrests in denton \u2019 s fracking protests tuesday ; only that mother also happens to be a great grandmother . ( published tuesday , june 16 , 2015 )     ( tue jun 16 16 : 40 : 54 pdt 2015 $ __output )     \" i will give vantage credit for this , \" said crowson . \" they ' ve owned it . they ' ve owned the responsibility for it . \"     through the course of their investigation , fire officials determined more than 42,000 gallons of fracking fluid escaped into the city ' s storm water system . they don ' t believe the city ' s water supply or the public are in any danger .     \" arlington environmental , the texas commission on environmental quality and the epa tested the water and soil samples in several locations and found pollutant material below levels that will require further remediation , \" said assistant fire chief jim self .     crowson said vantage has identified the faulty component that caused the well head to malfunction and has taken steps to fix the problem on that well and others .     the company has also paid the city more than $ 82,000 to reimburse taxpayers for the response to the leak . that doesn ' t include additional fines vantage had to pay for three citations the city issued following the incident .     \" our relationship with vantage has been very good historically , \" said crowson .     crowson did not mince words , though , when he discussed vantage ' s decision to wait nearly two hours before reporting the leak to 911. he called the company ' s actions \" unacceptable \" and said he ' s met with vantage and other energy companies to make sure they understand that .     \" call the fire department immediately , \" said crowson . \" that ' s been carefully and directly communicated . \"     in an email to nbc 5 , a spokesperson for vantage said , \" we ' ve worked collaboratively with the fire department on revised notification procedures . \"     crowson said the site has to pass one last inspection before drilling can resume . when that happens , he said the public will be notified . "
>>> Reflection.thought("I still need to contact Hannah. I will contact her now.")
>>> Enterprise.resolve_person("hannah")
{"user_id": "hannah", "email": "hannah@company.com", "full_name": "Hannah Jarvis"}
>>> Enterprise.send_message("hannah", "Hi Hannah, do you have any information about fracking fluid, Arlington, Texas, Vantage Energy or a gas leak?", title="Information Vantage Energy leak")
>>> System.finish()

# Event: Hannah Jarvis says: Sorry, I'm afraid I don't have any information on that topic.
>>> Reflection.thought("I have enough information to compile the summary. I will send a message to Alice, with the summary enclosed in [[[ brackets ]]].")
>>> Enterprise.send_message("alice", "Hi Alice, here's the summary I put together: [[[a massive leak of fracking fluid poured into the streets of arlington , texas , two months ago and forced the evacuation of a hundred homes . now city officials have taken vantage energy to task for its \" unacceptable \" handling of the 43,000-gallon spill , wfaa reports . during a city council meeting yesterday , it emerged that vantage had taken nearly two hours to call 911 despite the risk of a gas leak . \" this is unacceptable behavior , \" says an arlington city council member . according to fire chief don crowson , the two-hour delay was no joke : \" we \u2019 re not kidding around about the 911 issue , \" he tells the star-telegram during a break in city council . \" it could have ended in a bad outcome . two hours \u2019 advance notice could have helped a lot . \" still , officials say the environmental damage was not extensive and vantage has been cooperative . so what happened , exactly ? according to the city ' s report , a vantage well site sprung a leak on april 11 , which allowed fracking water and chemicals to boil up into arlington ' s streets , storm sewers , and streams . because the fracking fluid had been fracturing shale and freeing gas under high pressure , natural gas could have leaked at any time . now wfaa says a 1,500-gallon spill occurred at the same location a month before , and nbc-dallas/fort worth reports that the site is close to reopening . \" i just assumed this was a residential area and it would be free from industrial hazardous operations , \" says a resident after hearing about the earlier spill . \" now we see it ' s not . \"]]]", title="Vantage Energy leak summary")
>>> System.finish()

# Event: Alice Anastasiou says: Thanks! That's all I need.
>>> Enterprise.send_session_completed()
>>> System.finish()
"""


def _strip_reflection_thoughts(example: str) -> str:
    return "\n".join(
        [line for line in example.split("\n") if ">>> Reflection.thought" not in line]
    )


multinews_nocot_dialogue1 = _strip_reflection_thoughts(multinews_dialogue1)
multinews_nocot_dialogue2 = _strip_reflection_thoughts(multinews_dialogue2)
multinews_nocot_dialogue3 = _strip_reflection_thoughts(multinews_dialogue3)
