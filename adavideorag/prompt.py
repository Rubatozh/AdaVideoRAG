"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["entity_extraction"] = """-Goal-
Extract entities and directed relationships that are grounded in the supplied
video-segment text. The input contains global time ranges; preserve those ranges
instead of converting them to segment-local time.

-Entity schema-
For every entity, output exactly five fields:
("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<spatio_temporal_attribute>"{tuple_delimiter}"<entity_description>")

The spatio_temporal_attribute must contain the explicit global time range,
location, or scene supported by the input. Use UNKNOWN only when none is given.
Represent time-dependent actions as event/action entities when that is necessary
to express ordering without creating a self-edge on one person or object.

-Relationship schema-
For every relationship, output exactly six fields:
("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_type>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}<relationship_strength>)

Use a concise relationship_type such as BEFORE, AFTER, SIMULTANEOUS, OVERLAPS,
DURING, CAUSES, LOCATED_IN, or INTERACTS_WITH. Direction is semantic:
- source BEFORE target means the source occurs earlier than the target.
- source AFTER target means the source occurs later than the target.
- source CAUSES target means the source is the cause and the target is the effect.

Only create temporal or causal edges when the timestamped evidence supports
them. Do not infer an ordering merely from the order of sentences. Relationship
strength must be a number from 1 to 10.

-Output rules-
1. Use only these entity types: [{entity_types}].
2. Use {record_delimiter} between records.
3. End with {completion_delimiter}.
4. Return records only, in English, without Markdown or explanatory prose.

-Real Data-
Entity_types: [{entity_types}]
Text:
{input_text}

Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "action",
    "concept",
    "event",
    "location",
    "object",
    "organization",
    "person",
]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS[
    "query_rewrite"
] = """
-Goal-
Return retrieval requests for the question using exactly this JSON shape:
```json
{
  "caption": "a declarative visual-scene retrieval query, or null",
  "ASR": "a speech/subtitle retrieval query, or null",
  "OCR": ["up to five visible physical entities or locations"]
}
```
Use null for a modality that is not useful. OCR must contain physical entities
or locations rather than abstract concepts. Return the JSON object only.

## Example 1:
Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
```json
{
    "caption": "There have blue balloons over the long table in the middle of the room at the end of this video.",
    "ASR": "The location and the color of balloons, the number of the blue balloons.",
    "OCR": ["blue balloons", "long table"]
}
```
## Example 2:
Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
Your retrieve can be:
```json
{
    "caption": "A woman is on the right side of the man in black clothes.",
    "ASR": null,
    "OCR": ["the man in black", "woman"]
}
```
## Example 3:
Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
```json
{
    "caption": "The country recognized worldwide for its comedy.",
    "ASR": "The country recognized worldwide for its comedy.",
    "OCR": null
}
```
Return only the JSON retrieval request.

"""

PROMPTS["input"] = """
#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""





PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """-Goal-
For a given query, generate a declarative sentence to serve as a query for retrieving relevant knowledge.

######################
-Examples-
######################

Question: What are the main characters? \n(A) Alice\n(B) Bob\n(C) Charlie\n(D) Dana
################
Output:
The main characters. (Maybe Alice, Bob, Charlie or Dana)

Question: What locations are shown in the video?
################
Output:
The locations shown in the video.

Question: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
################
Output:
The animals that appear in the wildlife footage. (Maybe Lions, Elephants or Zebras)

#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""

PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """-Goal-
Given a question that may include scene-related information, generate a declarative sentence to serve as a query for retrieving relevant video segments.

######################
-Examples-
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Maybe Sunny, Rainy, Snowy or Windy)

#############################
-Real Data-
######################
Question: {input_text}
######################
Output:
"""



PROMPTS[
    "filtering_segment"
] = """---Role---
You are a helpful assistant to determine whether the video and  may contain information relevant to the knowledge based on its rough caption.
Please note that this is a rough caption of the video segments, which means it may not directly contain the answer but may indicate that the video segment is likely to contain information relevant to answering the question.

---Video Caption---

{caption}

---Knowledge We Need---

{knowledge}

---Answer---
Please provide an answer that begins with "yes" or "no",  followed by a brief step-by-step explanation.
Answer:
"""

PROMPTS[
    "adaptive_query"
] = """- Goal -
1. Given a query, Classify queries into exclusively one difficulty level (A/B/C) and  Crop command per output.
2. When you are confused about which level to choose, choose a higher level.

#### Classification Criteria ####
1. Level A (Local Perception or question is very easy that we can put all of video as input)
   - Scope: (1) Direct analysis of explicit timestamps. It is mainly used for local video perception, especially in the case of known frames or video segments, to understand the current frame or short-view frequency band. (e.g. "5s", "5s-6s", in the begining or in the end)
            Or the question is very easy, and there is no double or multiple logical relationship in the query. You can directly input all the videos to answer the query.

   - Action:
     * Example:
        Query: "What happens at 5s-6s?"
        Output: ###Level A###
     * Example:
        Query: "What do the performers wear?"
        Output: ###Level A###


2. Level B (Global Search)
   - Scope: It contains simple logical relationships. It can be used for global awareness, which requires global search and positioning of video segments related to questions,and then understanding of the searched video segments.
   - Action:
     * Example:
       Query: "Who received the most gifts, Amy or lucy?"
       Output: ###Level B###
     * Example:
       Query: "How many books when Tom left?"
       Output: ###Level B###
     * Example:
       Query: "At the beginning of the video, why does the woman in red change from long hair to short hair?"video_data: the  duration of video is 300 seconds
       Output: ###Level B###
     * Example:
       Query: "What is the little girl's expression after the performance?"
       Output: ###Level B###


3. Level C (Complex Reasoning)
   - Scope: Requires external knowledge/abstract interpretation. It may be necessary to make use of knowledge graphs of multi-layer or complex relationships,
   or semantic questions with high generalizations
   - Action:
     * Example:
       Query: "Analyze symbolic meaning of ocean shots"
       Output: ###Level C###
     * Example:
       Query: "At the end of the movie, what are the psychological changes of the protagonist?"  video_data: the  duration of video is 30 seconds
       Output: ###Level C###

###### OUTPUT FORMAT ########
Only respond with ONE of these, and the output needs to start with ### and end with ###

#############################
When you are confused about which level to choose, choose a higher level.
When the information is not clear, you can choose any level. Then, directly output a higher level.
There is no need to output the part about thinking.
#############################
"""
