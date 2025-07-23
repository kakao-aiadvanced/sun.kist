# streamlit

# 환경 변수 로드
try:
    from dotenv import load_dotenv

    load_dotenv()  # .env 파일이 있으면 로드
except ImportError:
    print("python-dotenv not installed, using system environment variables")

import streamlit as st
from sun_kist_agent import query_for_agent
import os

# 환경 변수 확인
print("streamlit - openai_key", os.getenv("OPENAI_API_KEY"))
print("streamlit - tavily_key", os.getenv("TAVILY_API_KEY"))


def main():

    st.set_page_config(page_title="Sun KIST AI Agent", page_icon="🤖", layout="wide")

    st.title("🤖 Sun KIST AI Agent")
    st.markdown("---")

    # 사이드바에 설정 정보
    with st.sidebar:
        st.header("⚙️ 설정 정보")
        st.info(
            """
          **이 에이전트는 다음 기능을 제공합니다:**
          
          1. 📄 문서 검색 (RAG)
          2. 🔍 관련성 체크
          3. 🌐 웹 검색 (Tavily)
          4. ✨ 답변 생성
          5. ✅ 환각 체크
          
          **참고 문서:**
          - LLM Agent 관련 문서
          - Prompt Engineering 가이드
          - LLM 공격 방법론
          """
        )

    # 메인 영역
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 질문하기")
        query = st.text_input(
            "질문을 입력하세요:", placeholder="예: where does messi play right now?"
        )
        if st.button("🚀 질문하기", type="primary", use_container_width=True):
            if not query.strip():
                st.error("질문을 입력해주세요!")
            else:
                with st.spinner("AI 에이전트가 답변을 생성하고 있습니다..."):
                    import time

                    # 진행상황 표시를 위한 placeholder
                    progress_placeholder = st.empty()

                    start_time = time.time()

                    try:
                        # 진행상황 콜백 함수 정의
                        def progress_callback(step_count, step_name):
                            progress_placeholder.text(
                                f"진행 중... 단계 {step_count}: {step_name}"
                            )

                        result = query_for_agent(query, progress_callback)

                        end_time = time.time()
                        processing_time = end_time - start_time

                        # 진행상황 메시지 제거
                        progress_placeholder.empty()

                        # 세션 상태 업데이트
                        if "query_count" not in st.session_state:
                            st.session_state.query_count = 0
                        if "success_count" not in st.session_state:
                            st.session_state.success_count = 0
                        if "query_history" not in st.session_state:
                            st.session_state.query_history = []

                        st.session_state.query_count += 1
                        st.session_state.query_history.append(query)

                        if result:
                            st.success(
                                f"✅ 답변 생성 완료! (소요시간: {processing_time:.2f}초)"
                            )

                            # 답변 표시
                            st.header("📝 답변")
                            if result.startswith("failed:"):
                                st.error(f"❌ 답변 생성에 실패했습니다: {result}")
                            else:
                                st.session_state.success_count += 1
                                st.markdown(result)

                                # 답변 평가
                                st.header("⭐ 답변 평가")
                                col_eval1, col_eval2, col_eval3 = st.columns(3)
                                with col_eval1:
                                    st.button(
                                        "👍 좋음", help="이 답변이 도움이 되었나요?"
                                    )
                                with col_eval2:
                                    st.button(
                                        "👎 나쁨",
                                        help="이 답변이 도움이 되지 않았나요?",
                                    )
                                with col_eval3:
                                    st.button(
                                        "🔄 다시 시도",
                                        help="같은 질문으로 다시 시도하시겠나요?",
                                    )
                        else:
                            st.error("❌ 답변을 생성할 수 없습니다.")

                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"❌ 오류가 발생했습니다: {str(e)}")

    with col2:
        st.header("🔍 결과")
        st.text_area("답변", placeholder="답변이 여기에 표시됩니다.")


if __name__ == "__main__":
    main()
