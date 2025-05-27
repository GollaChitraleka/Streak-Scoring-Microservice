# Action Type Specifications

## Login Action
- **Description**: Track user login events
- **Metadata Required**: None
- **Validation Criteria**: 
  - One valid login per UTC day
- **Streak Rules**: 
  - Consecutive UTC days with valid login
  - Grace period of 1 day allowed

## Quiz Action
- **Description**: Track quiz completion events
- **Metadata Required**: 
  - quiz_id (string): Unique identifier for the quiz
  - score (integer): User's score on the quiz
  - time_taken_sec (integer): Time taken to complete in seconds
- **Validation Criteria**:
  - Minimum score threshold (configurable, default: 5)
  - Maximum time threshold (configurable, default: 600 seconds)
  - One valid quiz per UTC day
- **Streak Rules**:
  - Consecutive UTC days with valid quiz completion
  - Grace period of 1 day allowed

## Help Post Action
- **Description**: Track help posts where users assist others
- **Metadata Required**:
  - content (string): The text content of the help post
  - word_count (integer): Number of words in the content
  - contains_code (boolean): Whether the post contains code
- **Validation Criteria**:
  - Minimum word count (configurable, default: 30 words)
  - Content quality validation via AI model
  - One valid help post per UTC day
- **Streak Rules**:
  - Consecutive UTC days with valid help post
  - Grace period of 1 day allowed
