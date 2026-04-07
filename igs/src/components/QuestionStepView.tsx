import type { QuestionChoice } from '../types/lesson'
import './QuestionStepView.css'

type QuestionStepViewProps = {
  questionText: string
  choices: QuestionChoice[]
  selectedChoiceId: string | null
  onSelectChoice: (choiceId: string) => void
  showQuestionButton: boolean
  onShowQuestion: () => void
}

export function QuestionStepView({
  questionText,
  choices,
  selectedChoiceId,
  onSelectChoice,
  showQuestionButton,
  onShowQuestion,
}: QuestionStepViewProps) {
  return (
    <>
      <div className="top-section">
        <div className="character-container">
          <div className="character-card" aria-hidden="true">
            •
          </div>
        </div>

        <div className="right-column">
          <h1 className="question-text">{questionText}</h1>
          {showQuestionButton && (
            <button type="button" className="show-question-button" onClick={onShowQuestion}>
              показать вопрос
            </button>
          )}
        </div>
      </div>

      <div className="answers-section">
        <section className="answers-panel" aria-label="Варианты ответа">
          {choices.map((choice) => {
            const isSelected = selectedChoiceId === choice.id
            return (
              <button
                key={choice.id}
                type="button"
                className={`choice-button ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectChoice(choice.id)}
              >
                {choice.text}
              </button>
            )
          })}
        </section>
      </div>
    </>
  )
}
