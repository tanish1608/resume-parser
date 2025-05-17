const { parseResume } = require('./resumeParser');

async function processResume() {
    const result = await parseResume('Resumee.pdf');
    console.log(result.sectionTexts);
}

processResume();