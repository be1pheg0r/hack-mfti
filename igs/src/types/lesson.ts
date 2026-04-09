export type img = string
export type video = string

export type QuestionChoice = {
  id: string
  text: string
  reaction: string
  isCorrect?: boolean
  image: img
}

export type QuestionStep = {
  kind: 'question'
  id: string
  question: string
  choices: QuestionChoice[]
  image: img
}

export type StoryStep = {
  kind: 'story'
  id: string
  text: string
  visualLabel: string
  image: img | video
}

export type LessonStep = QuestionStep | StoryStep
