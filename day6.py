import os
import json
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string"
                                    },
                                    "is_correct": {
                                        "type": "boolean"
                                    }
                                },
                                "required": ["answer", "is_correct"]
                            }
                        }
                    },
                    "required": ["question", "answers"]
                }
            }
        }
    }
}

prompt = PromptTemplate.from_template(
    "Make a quiz about {topic}. The difficulty level should be {difficulty}. Create questions that are appropriate for {difficulty} level."
    )

st.set_page_config(
    page_title="QuizGPT",
    page_icon="?",
)

st.title("QuizGPT")

@st.cache_data(show_spinner="Making quiz...")
def make_quiz(topic, difficulty):
    chain = prompt | llm
    return chain.invoke({"topic": topic, "difficulty": difficulty})

with st.sidebar:
    st.markdown("[🔗 Github Repo](https://github.com/hughqlee/fullstack_gpt_challenge)")  # 실제 주소로 변경 필요
    openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY"), type="password")

    if not openai_api_key:
        st.warning("OpenAI API Key를 입력하세요.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=openai_api_key
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions = [function]
    )

    topic = st.text_input("Enter a topic for the quiz")
    difficulty = st.selectbox(
        "Select difficulty level",
        ["easy", "medium", "hard"],
        index=1
    )
    
    if st.button("Generate Quiz"):
        quiz = make_quiz(topic, difficulty)
        st.session_state.quiz = json.loads(
            quiz.additional_kwargs["function_call"]["arguments"]
        )
        # 퀴즈 생성시 점수 상태 초기화
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}

if "quiz" in st.session_state:
    quiz_data = st.session_state.quiz
    
    if not st.session_state.get("quiz_submitted", False):
        with st.form("quiz_form"):
            user_answers = {}
            for i, question in enumerate(quiz_data["questions"]):
                st.write(f"**Question {i+1}:** {question['question']}")
                
                # 사용자가 이전에 선택한 답안이 있다면 그것을 기본값으로 사용
                previous_answer = st.session_state.user_answers.get(i, None)
                try:
                    default_index = [answer["answer"] for answer in question["answers"]].index(previous_answer) if previous_answer else None
                except ValueError:
                    default_index = None
                
                value = st.radio(
                    "Select the correct answer",
                    [answer["answer"] for answer in question["answers"]],
                    index=default_index,
                    key=f"question_{i}"
                )
                user_answers[i] = value
                
            if st.form_submit_button("Submit Quiz"):
                st.session_state.user_answers = user_answers
                st.session_state.quiz_submitted = True
                st.rerun()
    
    else:
        # 점수 계산 및 결과 표시
        correct_count = 0
        total_questions = len(quiz_data["questions"])
        
        for i, question in enumerate(quiz_data["questions"]):
            user_answer = st.session_state.user_answers.get(i)
            st.write(f"**Question {i+1}:** {question['question']}")
            
            # 정답 찾기
            correct_answer = None
            for answer in question["answers"]:
                if answer["is_correct"]:
                    correct_answer = answer["answer"]
                    break
            
            # 사용자 답안 표시
            if user_answer == correct_answer:
                st.success(f"✅ Your answer: {user_answer}")
                correct_count += 1
            else:
                st.error(f"❌ Your answer: {user_answer}")
                st.info(f"💡 Correct answer: {correct_answer}")
            
            st.write("---")
        
        # 최종 점수 표시
        score = correct_count / total_questions
        st.write(f"## 결과: {correct_count}/{total_questions} ({score:.1%})")
        
        # 만점인 경우 축하 메시지
        if score == 1.0:
            st.success("🎉 완벽합니다! 만점입니다!")
            st.balloons()
        else:
            st.warning(f"조금 더 공부해보세요! 점수: {score:.1%}")
            
            # 재시험 버튼
            if st.button("다시 시험 보기"):
                st.session_state.quiz_submitted = False
                st.rerun()

