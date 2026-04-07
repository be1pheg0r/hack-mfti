import { useMemo, useState } from 'react'
import './App.css'
import { LESSON_STEPS } from './data/lessons'
import { QuestionStepView } from './components/QuestionStepView'
import { StoryStepView } from './components/StoryStepView'

function App() {
  const [stepIndex, setStepIndex] = useState(0)
  const [selectedChoiceId, setSelectedChoiceId] = useState<string | null>(null)
  const [temporaryReaction, setTemporaryReaction] = useState<string | null>(null)

  const currentStep = LESSON_STEPS[stepIndex]
  const progressPercent = ((stepIndex + 1) / LESSON_STEPS.length) * 100

  const selectedChoice = useMemo(() => {
    if (!currentStep || currentStep.kind !== 'question' || !selectedChoiceId) {
      return null
    }
    return currentStep.choices.find((choice) => choice.id === selectedChoiceId) ?? null
  }, [currentStep, selectedChoiceId])

  const isContinueDisabled = currentStep.kind === 'question' && !selectedChoice

  const goToNextStep = () => {
    if (stepIndex >= LESSON_STEPS.length - 1) {
      setStepIndex(0)
      setSelectedChoiceId(null)
      setTemporaryReaction(null)
      return
    }
    setStepIndex((prev) => prev + 1)
    setSelectedChoiceId(null)
    setTemporaryReaction(null)
  }

  const handleContinue = () => {
    if (currentStep.kind === 'question' && !selectedChoice) {
      return
    }
    goToNextStep()
  }

  const handleSelectChoice = (choiceId: string) => {
    if (currentStep.kind !== 'question') {
      return
    }
    setSelectedChoiceId(choiceId)
    const choice = currentStep.choices.find((item) => item.id === choiceId)
    setTemporaryReaction(choice?.reaction ?? null)
  }

  return (
    <main className="lesson-page">
      <header className="lesson-header">
        <div className="progress-track" aria-label="Прогресс урока">
          <div
            className="progress-fill"
            style={{ width: `${progressPercent}%` }}
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progressPercent)}
          ></div>
        </div>
      </header>

      <section className={`lesson-layout ${currentStep.kind === 'story' ? 'story-layout' : ''}`}>
        {currentStep.kind === 'question' ? (
          <QuestionStepView
            questionText={temporaryReaction ?? currentStep.question}
            choices={currentStep.choices}
            selectedChoiceId={selectedChoiceId}
            onSelectChoice={handleSelectChoice}
            showQuestionButton={Boolean(temporaryReaction)}
            onShowQuestion={() => setTemporaryReaction(null)}
          />
        ) : (
          <StoryStepView step={currentStep} />
        )}
      </section>

      <footer className="lesson-footer">
        <button
          type="button"
          className="continue-button"
          onClick={handleContinue}
          disabled={isContinueDisabled}
        >
          Продолжить
        </button>
      </footer>
    </main>
  )
}

export default App
