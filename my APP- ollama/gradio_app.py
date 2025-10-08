import time
import gradio as gr

try:
    from chatbot import run_complete_rag_pipeline
except Exception:
    run_complete_rag_pipeline = None

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

#app { max-width: 760px; margin: 0 auto; }

/* Title & description */
#title {
  font-family: 'Poppins', 'Trebuchet MS', Arial, sans-serif;
  text-align: center;
  font-size: 52px;
  font-weight: 700;
  letter-spacing: .5px;
  margin: 28px 0 8px;
}

#desc  { text-align: center; font-size: 18px; margin: 0 20px 28px; opacity: .9; }

/* Hide labels */
[data-testid="block-label"] { display: none !important; }
#msg_input [data-testid="block-label"] { display: none !important; }
label[for="msg_input"] { display: none !important; }

/* Rounded chat bubbles + fun font */
[data-testid="chatbot"] .message,
.chat-message,
.message {
  font-family: 'Comic Neue','Comic Sans MS','Trebuchet MS', Arial, sans-serif !important;
  border-radius: 18px !important;
}

/* --- Remove the X / clear button in the textbox --- */
#msg_input input::-ms-clear,
#msg_input input::-ms-reveal,
#msg_input input::-webkit-search-cancel-button,
#msg_input textarea::-webkit-search-cancel-button {
  display: none !important;
  -webkit-appearance: none !important;
}
#msg_input textarea {
  background-image: none !important;
  caret-color: white;
}

/* --- Typing dots bubble --- */
.typing {
  display: inline-block; letter-spacing: .15em; opacity: .85;
}
.typing .dot {
  font-size: 14px;           /* smaller dots */
  animation: blink 1.2s infinite;
}
.typing .dot:nth-child(2) { animation-delay: .2s; }
.typing .dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,20%{opacity:.25;}50%{opacity:1;}100%{opacity:.25;} }
"""

def _stream_text(answer: str, delay: float = 0.02):
    buf = []
    for token in answer.split():
        buf.append(token)
        if len(buf) >= 3:
            yield " ".join(buf) + " "
            time.sleep(delay)
            buf = []
    if buf:
        yield " ".join(buf)

def _typing_html():
    return '<span class="typing"><span class="dot">●</span><span class="dot">●</span><span class="dot">●</span></span>'

def respond(message, history):
    if history is None:
        history = []

    history = history + [(message, _typing_html())]
    yield gr.update(value=""), history, gr.update(interactive=False), gr.update(interactive=False)

    try:
        full_answer = run_complete_rag_pipeline(message) if callable(run_complete_rag_pipeline) else f"Echo: {message}"
    except Exception as e:
        full_answer = f"Error: {e}"

    streamed = ""
    for chunk in _stream_text(full_answer):
        streamed += chunk
        history[-1] = (message, streamed)
        yield gr.update(value=""), history, gr.update(interactive=False), gr.update(interactive=False)

    history[-1] = (message, full_answer)
    yield gr.update(value=""), history, gr.update(interactive=True), gr.update(interactive=True)

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), elem_id="app") as demo:
    gr.Markdown("<div id='title'>CS CheatSheet BOT</div>")
    gr.Markdown("""
        <div id='desc' style='text-align:center;'>
        Every syllabus, catalog, and course secret in one place<br>
        <span style='display:block; text-align:center;'>because being a CS student is already hard enough</span>
        </div>
                """)

    chatbot = gr.Chatbot(
        show_label=False,
        height=460,
        bubble_full_width=False,
        show_copy_button=False
    )

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Type your message…",
            label=None,
            show_label=False,
            elem_id="msg_input",
            scale=8
        )
        send = gr.Button("Send", scale=1)

    txt.submit(respond, [txt, chatbot], [txt, chatbot, txt, send])
    send.click(respond, [txt, chatbot], [txt, chatbot, txt, send])

demo.launch()
