import json
from typing import Dict, List, Any

concept_extraction_template = """You are given a conversation in JSON format. Each element of the JSON array is a dictionary (object) that represents a single turn in the conversation. Each dictionary has two keys:
1. **role** - which can be "user" or "assistant"  
2. **content** - the text content of that turn

Your task:  

1. Read through each turn in the conversation.  
2. Extract all **related concepts** from each turn's content. A concept is defined as any short phrase (one to a few words) that represents an idea, topic, event, name, or domain-specific term mentioned in that turn.  
   - **Examples of potential concepts** might include specific events ("Trump Inauguration"), general topics ("climate change"), ideas ("economic resilience"), niche terms ("public-benefit corporation"), or anything else that stands out.  
   - **Have a low bar for inclusion**; if something seems relevant, it can be included as a concept.  
3. Separate the extracted concepts based on whether they came from the **user**'s messages or the **assistant**'s messages.  
4. Return a JSON dict containing four lists:
   - `user_concepts_explicit`: all unique concepts explicitly mentioned in the user's messages (as short strings)
   - `assistant_concepts_explicit`: all unique concepts explicitly mentioned in the assistant's messages (as short strings)
   - `user_concepts_related`: all unique concepts related to the user's messages (as short strings)
   - `assistant_concepts_related`: all unique concepts related to the assistant's messages (as short strings)

Important details:
- If the user or the assistant repeats a concept multiple times, list it only once in the respective output array.  
- Each concept should be a **distinct short phrase**.  
- Output **only** the JSON object and nothing else (no extra commentary or explanation).
- Always give concepts in English, even if the conversation is in another language.

### Input 1

```json
[
  {
    "content": "What is reality shifting?",
    "role": "user"
  },
  {
    "content": "Reality shifting is a practice or belief in which individuals attempt to shift their consciousness or awareness to enter and experience alternate realities or dimensions. It is often associated with the idea that individuals have the ability to consciously control their experiences and perceive different realities. Reality shifting can involve various techniques, such as guided meditations, visualization exercises, or specific rituals, with the aim of accessing different planes of existence, fictional worlds, or desired scenarios. Some individuals perceive reality shifting as a means to escape or explore possibilities beyond their current physical reality. However, it is important to note that reality shifting is generally considered a subjective experience and is not scientifically proven.",
    "role": "assistant"
  },
  {
    "content": "Has any government ever studied reality shifting?",
    "role": "user"
  },
  {
    "content": "There is no evidence or official documentation to suggest that any government has conducted serious or official studies on reality shifting. The concept of reality shifting lies within the realm of spirituality, metaphysics, and personal belief systems rather than scientific inquiry or government research. Governments typically prioritize funding and conducting studies related to more tangible and empirically verifiable subjects, such as medicine, technology, economics, and social sciences. Consequently, reality shifting remains largely within the sphere of personal exploration and subjective experiences rather than being subject to government study.",
    "role": "assistant"
  }
]
```

### Output 1 (Concepts in English)

```json
{
    "user_concepts_explicit": ["reality shifting", "government"],
    "user_concepts_related": ["conspiracy theory", "government research", "government conspiracy", "spirituality", "supernatural phenomenon"],
    "assistant_concepts_explicit": ["reality shifting", "consciousness", "awareness", "alternate reality", "alternate dimension", "meditation", "guided meditation", "visualization exercise", "ritual", "plane of existence", "fictional world", "physical reality", "subjective experience", "scientific proof", "official documentation", "spirituality", "metaphysics", "personal belief system", "scientific inquiry", "government research", "government funding", "medicine", "technology", "economics", "social science", "subjective experience", "government study"],
    "assistant_concepts_related": ["pseudoscience", "conspiracy theory", "scientific skepticism", "government conspiracy", "supernatural phenomenon", "esotericism", "philosophy of consciousness", "parapsychology"]
}
```

### Input 2

```json
[
  {
    "content": "孕检需要做什么准备",
    "role": "user"
  },
  {
    "content": "孕检（产前检查）是为了评估孕妇及胎儿的健康状况，预防和诊断并发症、异常情况，制定孕产期的诊治计划和预防措施。孕检需要做以下准备工作：\n\n1. 登记个人信息：准备相关的个人信息，如年龄、职业、身份证、医保卡、电话等，方便登记建档。\n\n2. 预约检查时间：尽量选择孕早期，孕6~8周，进行第一次孕检并了解医院产检的流程以及接下来的检查计划。\n\n3. 了解家族遗传病史：掌握自己和配偶的家族病史，如是否存在遗传性疾病等，以便及时采取干预措施。\n\n4. 以往妊娠史：准备好孕产经历的详细情况，如胎数、产次、生育方式、孕期经历等。\n\n5. 了解检查项目：了解孕检的基本项目，例如血常规、尿常规、肝肾功能、血型、心电图等，以及相关的检查注意事项。\n\n6. 空腹检查：如需进行血液、尿液检查，要求空腹进行，当天请勿喝酒、刷牙或进食。\n\n7. 穿着舒适的衣物：孕检时可能需要做B超或体检，建议穿着宽松舒适的衣物，以便进行检查。\n\n8. 安排好休息时间：孕检流程较长，建议孕妇带上水和小零食，以防过长时间等待，产生低血糖等问题。\n\n9. 配偶陪同：建议配偶陪同前往孕检，分享孕育体验，共同关注胎儿和孕妇的健康。\n\n10. 注意心态：孕期是特殊的阶段，要保持良好心态，相信医生和自己，为胎儿的健康和安全出生创造良好的环境。",
    "role": "assistant"
  },
  {
    "content": "朝阳区妇幼保健院孕检材料",
    "role": "user"
  },
  {
    "content": "在中国北京市朝阳区妇幼保健院进行孕检需要准备以下资料和材料：\n\n1. 居民身份证：夫妻双方的居民身份证原件及复印件。如果身份证丢失或损坏，请携带暂住证或户口本代替。\n\n2. 结婚证：夫妻双方的结婚证原件及复印件。如果没有结婚证，可根据当地政策提供相关证明材料。\n\n3. 产检手册：建议带上前期的孕检结果和产检手册，以便于医生了解孕妇的孕期情况。\n\n4. 医保卡：携带医保卡有助于享受相应的医保待遇。\n\n5. 联系方式：准备好有效的联系电话，以便医院与孕妇（夫）联系。\n\n进行孕检时，请穿着舒适、宽松的衣物，以便于体检。建议带上足够的水和小零食，以防长时间等待。\n\n建议你在孕早期（孕6~8周）开始进行孕检，并在医院建立孕检档案。此外，尽量定期进行孕检，以便了解胎儿发育情况、预防并发症，并制定适当的诊治计划。\n\n注意：具体过程与材料可能因医院政策而有所不同，请根据当地实际情况来准备，并在孕检前与医院联系了解详细信息。",
    "role": "assistant"
  }
]
```

### Output 2 (Concepts in English)

```json
{
    "user_concepts_explicit": ["prenatal checkup", "Chaoyang District Maternal and Child Health Hospital"],
    "user_concepts_related": ["prenatal examination materials", "hospital policies", "maternal health services", "medical advice", "childbirth", "pregnancy", "motherhood", "Beijing healthcare system", "public health services"],
    "assistant_concepts_explicit": ["prenatal checkup", "health assessment", "complications", "abnormal conditions", "treatment plan", "personal information", "family medical history", "genetic disorders", "pregnancy history", "blood test", "urine test", "liver and kidney function", "blood type", "electrocardiogram", "ultrasound", "B-mode ultrasound", "fasting test", "comfortable clothing", "low blood sugar", "emotional wellbeing", "birth safety", "national ID card", "marriage certificate", "prenatal record booklet", "health insurance card", "contact information", "Chaoyang District Maternal and Child Health Hospital", "Chaoyang District"], 
    "assistant_concepts_related": ["maternal care", "prenatal health", "early pregnancy", "prenatal examination process", "hospital-specific requirements", "maternal wellbeing", "child health assessment", "preventive measure", "regional hospital policies", "Beijing healthcare system", "Beijing", "public health services"]
}
```

### Input 3

```json
{conversation}
```

### Output 3 (Concepts in English)

[FILL IN YOUR ANSWER HERE]"""


def fill_template_single(template: str, **kwargs: Dict[str, Any]) -> str:
    for key in kwargs:
        if not isinstance(kwargs[key], str):
            kwargs[key] = json.dumps(kwargs[key], indent=2)
    
    return template.format(**kwargs)

def fill_template_parallel(template: str, **kwargs: Dict[str, List[Any]]) -> List[str]:
    list_length = len(next(iter(kwargs.values())))
    for key in kwargs:
        if not all(isinstance(value, str) for value in kwargs[key]):
            assert len(kwargs[key]) == list_length, f"Length of {key} does not match the length of other lists."
            kwargs[key] = [json.dumps(value, indent=2) for value in kwargs[key]]
    
    return [template.format(**{key: values[i] for key, values in kwargs.items()}) for i in range(list_length)]