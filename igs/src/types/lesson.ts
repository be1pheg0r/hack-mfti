export type QuestionChoice = {
  id: string
  text: string
  reaction: string
}

export type QuestionStep = {
  kind: 'question'
  id: string
  question: string
  choices: QuestionChoice[]
}

export type StoryStep = {
  kind: 'story'
  id: string
  text: string
  visualLabel: string
}

export type LessonStep = QuestionStep | StoryStep
